job "agent-ops-nightly-report" {
  datacenters = ["dc1"]
  namespace   = "agents"
  type        = "batch"

  meta {
    managed_by   = "cat-agent"
    team         = "ops"
    agent        = "nightly-report"
    trigger      = "schedule"
    jobs_mode    = "inline"
    image_tag    = "ops/nightly-report:abc"
    manifest_sha = "cafe"
    deployed_by  = "tester"
    deployed_at  = "2026-01-01T00:00:00Z"
  }

  periodic {
    cron             = "0 */5 * * *"
    time_zone        = "Europe/Istanbul"
    prohibit_overlap = true
  }

  group "agent" {
    restart {
      attempts = 1
      mode     = "fail"
    }

    task "agent" {
      driver = "docker"

      config {
        image = "ops/nightly-report:abc"
        # force_pull must stay false (Nomad default). With registry=local the
        # image exists only on the host Docker daemon shared with this client;
        # force_pull=true would try to pull a tag that exists nowhere. With a
        # remote registry and content-addressed tags the image is pulled once;
        # forcing a pull every restart only adds latency and a failure mode.
        readonly_rootfs = true
        command = "cat-agent"
        args = [
          "serve",
          "--factory", "report:registry",
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
        CAT_AGENT_ENTRYPOINT = "report:registry"
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
