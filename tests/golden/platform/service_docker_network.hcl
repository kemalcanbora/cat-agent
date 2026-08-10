job "agent-growth-funding-scout" {
  datacenters = ["dc1"]
  namespace   = "agents"
  type        = "service"

  meta {
    managed_by   = "cat-agent"
    team         = "growth"
    agent        = "funding-scout"
    trigger      = "http"
    jobs_mode    = "inline"
    image_tag    = "growth/funding-scout:abc"
    manifest_sha = "deadbeef"
    deployed_by  = "tester"
    deployed_at  = "2026-01-01T00:00:00Z"
  }

  update {
    max_parallel      = 1
    min_healthy_time  = "10s"
    healthy_deadline  = "5m"
    progress_deadline = "600s"
    auto_revert       = true
  }

  reschedule {
    delay          = "30s"
    delay_function = "exponential"
    max_delay      = "1h"
    unlimited      = true
  }

  group "agent" {
    count = 1

    network {
      # DOCKER DESKTOP / macOS ONLY: join a user Docker network. Bridge CNI needs
      # host netns mounts Docker Desktop cannot provide. On Linux: omit
      # docker_network in platform config so this branch uses mode = "bridge".
      port "http" {
        to = 8080
      }
    }

    restart {
      attempts = 2
      mode     = "fail"
    }

    task "agent" {
      driver = "docker"

      config {
        image = "growth/funding-scout:abc"
        # force_pull must stay false (Nomad default). With registry=local the
        # image exists only on the host Docker daemon shared with this client;
        # force_pull=true would try to pull a tag that exists nowhere. With a
        # remote registry and content-addressed tags the image is pulled once;
        # forcing a pull every restart only adds latency and a failure mode.
        # DOCKER DESKTOP / macOS ONLY: network_mode = compose network (no host
        # port publish — Desktop cannot bind Nomad's advertise IP). Reach via
        # shared Docker network + address_mode=driver (or Traefik on the host).
        # On Linux: omit docker_network; use ports = ["http"] under bridge CNI.
        network_mode = "nomad_deploy_hashicorp"
        # DOCKER DESKTOP / macOS ONLY: pinned Consul DNS IP (compose). Resolve
        # *.service.consul inside the alloc — Docker DNS alone cannot. On Linux:
        # set consul_dns to the real Consul DNS address (not 10.32.0.2).
        dns_servers = ["10.32.0.2"]
        readonly_rootfs = true
        command = "cat-agent"
        args = [
          "serve",
          "--factory", "scout:registry",
          "--host", "0.0.0.0",
          "--port", "8080",
        ]
      }

      # Task-level service required for address_mode=driver (container IP on
      # the shared Docker network is reachable from Consul).
      service {
        name         = "agent-growth-funding-scout"
        port         = "http"
        provider     = "consul"
        address_mode = "driver"
        # Traefik (Consul Catalog): Host from platform.ingress_host_template
        # (default {team}-{name}.localhost). Match public_url_template for humans.
        tags = [
          "traefik.enable=true",
          "traefik.http.routers.agent-growth-funding-scout.rule=Host(`growth-funding-scout.localhost`)",
          "traefik.http.routers.agent-growth-funding-scout.entrypoints=web",
          "traefik.http.services.agent-growth-funding-scout.loadbalancer.server.port=8080",
        ]

        check {
          name         = "healthz"
          type         = "http"
          path         = "/healthz"
          interval     = "10s"
          timeout      = "2s"
          address_mode = "driver"
        }

        check {
          name         = "readyz"
          type         = "http"
          path         = "/readyz"
          interval     = "15s"
          timeout      = "3s"
          address_mode = "driver"
          check_restart {
            limit = 3
            grace = "30s"
          }
        }
      }

      resources {
        cpu    = 500
        memory = 512
      }

      kill_timeout = "315s"

      vault {
        policies = [
          "cat-agent-llm-growth",
        ]
      }

      # Team LiteLLM virtual key — never put OPENAI_API_KEY in env {} below.
      template {
        destination = "secrets/llm.env"
        env         = true
        change_mode = "restart"
        data = <<EOF
OPENAI_API_KEY={{ with secret "secret/data/platform/llm/teams/growth" }}{{ .Data.data.api_key }}{{ end }}
EOF
      }

      env {
        CAT_AGENT_MANAGED = "1"
        CAT_AGENT_ENTRYPOINT = "scout:registry"
        CAT_AGENT_MODE = "service"
        OPENAI_BASE_URL = "http://llm-gateway.service.consul:4000/v1"
        CAT_AGENT_LLM_BASE_URL = "http://llm-gateway.service.consul:4000/v1"
        CAT_AGENT_TOOLS_ALLOW = "storage,web_search"
        CAT_AGENT_SERVE_HOST = "0.0.0.0"
        CAT_AGENT_SERVE_PORT = "8080"
        PORT = "8080"
        LOG_LEVEL = "INFO"
        CAT_AGENT_LLM_MODEL = "default"
        CAT_AGENT_MODEL_ALIAS = "default"
        CAT_AGENT_MAX_CONCURRENCY = "8"
        CAT_AGENT_MAX_QUEUE = "8"
        CAT_AGENT_SHUTDOWN_TIMEOUT = "45"
        CAT_AGENT_RUN_TIMEOUT = "300"
      }
    }
  }
}
