job "agent-ops-queue-worker" {
  datacenters = ["dc1"]
  namespace   = "agents"
  type        = "service"

  meta {
    managed_by   = "cat-agent"
    team         = "ops"
    agent        = "queue-worker"
    trigger      = "worker"
    jobs_mode    = "inline"
    image_tag    = "ops/queue-worker:abc"
    manifest_sha = "babe"
    deployed_by  = "tester"
    deployed_at  = "2026-01-01T00:00:00Z"
  }

  group "agent" {
    count = 1

    restart {
      attempts = 2
      mode     = "fail"
    }

    # Nomad only knows the process is alive, not that it is consuming.
    # Emit heartbeats through observability handlers and alert on their absence.
    task "agent" {
      driver = "docker"

      config {
        image = "ops/queue-worker:abc"
        # force_pull must stay false (Nomad default). With registry=local the
        # image exists only on the host Docker daemon shared with this client;
        # force_pull=true would try to pull a tag that exists nowhere. With a
        # remote registry and content-addressed tags the image is pulled once;
        # forcing a pull every restart only adds latency and a failure mode.
        readonly_rootfs = true
        command = "cat-agent"
        args = [
          "serve",
          "--factory", "worker:registry",
          "--host", "0.0.0.0",
          "--port", "8080",
        ]
      }

      resources {
        cpu    = 500
        memory = 512
      }

      kill_timeout = "315s"

      vault {
        policies = [
          "cat-agent-llm-ops",
        ]
      }

      # Team LiteLLM virtual key — never put OPENAI_API_KEY in env {} below.
      template {
        destination = "secrets/llm.env"
        env         = true
        change_mode = "restart"
        data = <<EOF
OPENAI_API_KEY={{ with secret "secret/data/platform/llm/teams/ops" }}{{ .Data.data.api_key }}{{ end }}
EOF
      }

      env {
        CAT_AGENT_MANAGED = "1"
        CAT_AGENT_ENTRYPOINT = "worker:registry"
        CAT_AGENT_MODE = "service"
        OPENAI_BASE_URL = "http://llm-gateway.service.consul:4000/v1"
        CAT_AGENT_LLM_BASE_URL = "http://llm-gateway.service.consul:4000/v1"
        CAT_AGENT_TOOLS_ALLOW = ""
        CAT_AGENT_SERVE_HOST = "0.0.0.0"
        CAT_AGENT_SERVE_PORT = "8080"
        PORT = "8080"
        CAT_AGENT_LLM_MODEL = "default"
        CAT_AGENT_MODEL_ALIAS = "default"
      }
    }
  }
}
