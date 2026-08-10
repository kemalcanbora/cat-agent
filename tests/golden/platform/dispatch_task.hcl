job "agent-growth-heavy-scout-task" {
  datacenters = ["dc1"]
  namespace   = "agents"
  type        = "batch"

  meta {
    managed_by   = "cat-agent"
    team         = "growth"
    agent        = "heavy-scout"
    trigger      = "http"
    jobs_mode    = "dispatch"
    image_tag    = "growth/heavy-scout:abc"
    manifest_sha = "face"
    deployed_by  = "tester"
    deployed_at  = "2026-01-01T00:00:00Z"
  }

  parameterized {
    payload       = "required"
    meta_required = ["job_id", "requested_by"]
  }

  group "task" {
    restart {
      attempts = 0
      mode     = "fail"
    }

    task "task" {
      driver = "docker"

      config {
        image = "growth/heavy-scout:abc"
        # force_pull must stay false (Nomad default). With registry=local the
        # image exists only on the host Docker daemon shared with this client;
        # force_pull=true would try to pull a tag that exists nowhere. With a
        # remote registry and content-addressed tags the image is pulled once;
        # forcing a pull every restart only adds latency and a failure mode.
        readonly_rootfs = true
        command = "python"
        args = ["-m", "cat_agent.serve.task"]
      }

      resources {
        cpu    = 500
        memory = 512
      }

      kill_timeout = "1800s"

      vault {
        policies = [
          "cat-agent-llm-growth",
        ]
      }

      # Team LiteLLM virtual key — never put OPENAI_API_KEY in env {} below.
      template {
        destination = "secrets/llm.env"
        env         = true
        change_mode = "noop"
        data = <<EOF
OPENAI_API_KEY={{ with secret "secret/data/platform/llm/teams/growth" }}{{ .Data.data.api_key }}{{ end }}
EOF
      }

      env {
        CAT_AGENT_MANAGED = "1"
        CAT_AGENT_ENTRYPOINT = "scout:registry"
        CAT_AGENT_MODE = "task"
        OPENAI_BASE_URL = "http://llm-gateway.service.consul:4000/v1"
        CAT_AGENT_LLM_BASE_URL = "http://llm-gateway.service.consul:4000/v1"
        CAT_AGENT_TOOLS_ALLOW = ""
        CAT_AGENT_SERVE_HOST = "0.0.0.0"
        CAT_AGENT_SERVE_PORT = "8080"
        PORT = "8080"
        CAT_AGENT_MODE = "task"
        CAT_AGENT_JOB_ID = "${NOMAD_META_job_id}"
        CAT_AGENT_TRACE_ID = "${NOMAD_META_requested_by}"
        CAT_AGENT_PAYLOAD = "${NOMAD_TASK_DIR}/payload"
        CAT_AGENT_LLM_MODEL = "default"
        CAT_AGENT_MODEL_ALIAS = "default"
      }
    }
  }
}
