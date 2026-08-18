# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build (and optionally push) agent container images."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.manifest import AgentManifest

RunFn = Callable[[Sequence[str], Optional[Path]], subprocess.CompletedProcess]
LoginFn = Callable[[], subprocess.CompletedProcess]


class BuildError(RuntimeError):
    """Image build or import check failed."""


def _default_run(cmd: Sequence[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def find_repo_root(start: Path) -> Optional[Path]:
    for p in [start.resolve(), *start.resolve().parents]:
        if (p / 'pyproject.toml').is_file() and (p / 'cat_agent').is_dir():
            return p
    return None


def short_content_tag(
    manifest: AgentManifest,
    source_dir: Path,
    *,
    config: Optional[PlatformConfig] = None,
) -> str:
    """Content-addressed local tag: ``{team}/{name}:{hash12}``.

    Local mode hashes monorepo ``cat_agent`` plus the example (both are COPY'd).
    Remote mode hashes only the example — the framework comes from the base
    image, so a source-tree framework change must not silently retag the image.
    """
    h = hashlib.sha256()
    h.update(manifest.job_id().encode())

    def _feed(root: Path, *, relative_to: Path) -> None:
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            if path.name == '.env' or '.git' in path.parts or '__pycache__' in path.parts:
                continue
            if path.suffix in {'.so', '.dylib', '.pyc'}:
                continue
            h.update(path.relative_to(relative_to).as_posix().encode())
            h.update(path.read_bytes()[:4096])

    _feed(source_dir, relative_to=source_dir)
    include_framework = config is None or config.is_local_registry()
    if include_framework:
        repo = find_repo_root(source_dir)
        if repo is not None:
            pkg = repo / 'cat_agent'
            if pkg.is_dir():
                _feed(pkg, relative_to=repo)
    return PlatformConfig.local_image_tag(
        manifest.team, manifest.name, h.hexdigest()[:12]
    )


def _write_local_monorepo_dockerfile(
    dest: Path,
    *,
    rel_example: str,
    entrypoint: str,
    base_image: str,
) -> None:
    """Local-registry Dockerfile: COPY source-tree cat_agent (monorepo iterate)."""
    dest.write_text(
        textwrap.dedent(
            f"""\
            FROM {base_image}
            RUN pip install --no-cache-dir \\
                  fastapi uvicorn httpx pydantic openai python-dotenv loguru \\
                  tiktoken pillow json5 jsonlines jsonschema docstring_parser \\
                  eval_type_backport cryptography keyring requests pyyaml sqlalchemy
            WORKDIR /opt/cat-agent
            COPY cat_agent ./cat_agent
            COPY {rel_example} /app
            RUN printf '%s\\n' \\
                  '#!/usr/bin/env python' \\
                  'import sys' \\
                  'from cat_agent.cli import main' \\
                  'sys.exit(main())' \\
                  > /usr/local/bin/cat-agent && chmod +x /usr/local/bin/cat-agent
            ENV PYTHONPATH=/opt/cat-agent:/app
            ENV CAT_AGENT_ENTRYPOINT={entrypoint}
            ENV CAT_AGENT_MANAGED=1
            WORKDIR /app
            RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
            """
        ),
        encoding='utf-8',
    )


def _write_remote_dockerfile(
    dest: Path,
    *,
    rel_example: str,
    entrypoint: str,
    base_image: str,
) -> None:
    """Remote-registry Dockerfile: example only — framework from the base image.

    Do NOT COPY cat_agent here. A source-tree COPY would shadow the pip-installed
    package in the base image and produce two different frameworks depending on
    how the agent was deployed. Remote builds require ``build-base`` (or an
    equivalent base that already has cat-agent installed).
    """
    dest.write_text(
        textwrap.dedent(
            f"""\
            FROM {base_image}
            WORKDIR /app
            COPY {rel_example} /app
            RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
            ENV PYTHONPATH=/app
            ENV CAT_AGENT_ENTRYPOINT={entrypoint}
            ENV CAT_AGENT_MANAGED=1
            """
        ),
        encoding='utf-8',
    )


def stage_agent_build_context(
    manifest: AgentManifest,
    config: PlatformConfig,
    source_dir: Path,
    ctx: Path,
) -> Path:
    """Stage the Docker build context into *ctx*; return the Dockerfile path.

    Local registry: copy monorepo ``cat_agent`` + example (COPY in Dockerfile).
    Remote registry: copy example only — framework must come from ``base_image``.
    """
    source_dir = source_dir.resolve()
    repo = find_repo_root(source_dir) or find_repo_root(Path.cwd())
    local = config.is_local_registry()

    if repo is not None:
        try:
            rel = source_dir.relative_to(repo).as_posix()
        except ValueError:
            # source_dir is outside the repo (e.g. mktemp copy for a second
            # team). Stage the example as ./example.
            rel = 'example'
        if local:
            # LOCAL only: source-tree cat_agent. Remote must not COPY it — that
            # would shadow the pip-installed framework in the base image.
            shutil.copytree(
                repo / 'cat_agent',
                ctx / 'cat_agent',
                ignore=shutil.ignore_patterns('__pycache__', '*.so', '.env'),
            )
            # Linux containers cannot load a macOS .so — ship a tiny stub so
            # framework imports succeed; RAG tools still need a real wheel.
            import platform as py_platform

            host_so = repo / 'cat_agent' / '_native.abi3.so'
            if py_platform.system() == 'Linux' and host_so.is_file():
                shutil.copy2(host_so, ctx / 'cat_agent' / '_native.abi3.so')
            else:
                (ctx / 'cat_agent' / '_native.py').write_text(
                    textwrap.dedent(
                        '''\
                        """Minimal stub for container builds without a Linux native wheel."""
                        WORDS_TO_IGNORE = frozenset()

                        def clean_en_token(token: str) -> str:
                            return token

                        def tokenize_and_filter(text: str):
                            return [t for t in text.split() if t]

                        def split_text_into_keywords(text: str):
                            return tokenize_and_filter(text)

                        def string_tokenizer(text: str):
                            return tokenize_and_filter(text)

                        def init_qwen_tokenizer(path: str = "") -> None:
                            return None

                        def count_qwen_tokens(text: str) -> int:
                            return max(1, len(text or "") // 4)

                        def truncate_qwen_text(
                            text: str, max_token: int, keep_both_sides: bool = False
                        ) -> str:
                            approx = max(1, int(max_token) * 4)
                            text = text or ""
                            if len(text) <= approx:
                                return text
                            if keep_both_sides:
                                half = approx // 2
                                return text[:half] + text[-half:]
                            return text[:approx]

                        def truncate_messages(messages, max_tokens: int):
                            msgs = [dict(m) for m in (messages or [])]
                            budget = max(1, int(max_tokens))

                            def _tok(m):
                                return count_qwen_tokens(str(m.get("text") or ""))

                            total = sum(_tok(m) for m in msgs)
                            if total <= budget:
                                return msgs
                            out = list(msgs)
                            while len(out) > 1 and sum(_tok(m) for m in out) > budget:
                                drop_at = 0
                                if out and str(out[0].get("role") or "") == "system":
                                    drop_at = 1 if len(out) > 1 else 0
                                if drop_at >= len(out):
                                    break
                                out.pop(drop_at)
                            if out and sum(_tok(m) for m in out) > budget:
                                last = dict(out[-1])
                                last["text"] = truncate_qwen_text(
                                    str(last.get("text") or ""), budget
                                )
                                out[-1] = last
                            return out

                        class RagIndex:
                            def __init__(self, *a, **k):
                                raise RuntimeError("native RagIndex unavailable in this image")

                        class VectorIndex:
                            def __init__(self, *a, **k):
                                raise RuntimeError("native VectorIndex unavailable in this image")

                        def split_doc_to_chunks(*a, **k):
                            raise RuntimeError("native split_doc_to_chunks unavailable")

                        def hash_embed(*a, **k):
                            raise RuntimeError("native hash_embed unavailable")
                        '''
                    ),
                    encoding='utf-8',
                )
        shutil.copytree(
            source_dir,
            ctx / rel,
            ignore=shutil.ignore_patterns('.env', '__pycache__', '.git'),
        )
        dockerfile = ctx / 'Dockerfile'
        if local:
            _write_local_monorepo_dockerfile(
                dockerfile,
                rel_example=rel,
                entrypoint=manifest.runtime.entrypoint,
                base_image=config.base_image,
            )
        else:
            _write_remote_dockerfile(
                dockerfile,
                rel_example=rel,
                entrypoint=manifest.runtime.entrypoint,
                base_image=config.base_image,
            )
        return dockerfile

    for item in source_dir.iterdir():
        if item.name == '.env':
            continue
        dest = ctx / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                dest,
                ignore=shutil.ignore_patterns('.env', '__pycache__', '.git'),
            )
        else:
            shutil.copy2(item, dest)
    dockerfile = ctx / 'Dockerfile'
    dockerfile.write_text(
        textwrap.dedent(
            f"""\
            FROM {config.base_image}
            WORKDIR /app
            COPY . /app
            RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
            ENV PYTHONPATH=/app
            ENV CAT_AGENT_ENTRYPOINT={manifest.runtime.entrypoint}
            ENV CAT_AGENT_MANAGED=1
            """
        ),
        encoding='utf-8',
    )
    return dockerfile


def registry_host(config: PlatformConfig) -> str:
    """Host:port (or host) used for ``docker login`` — first path segment of registry."""
    return config.registry.strip().rstrip('/').split('/', 1)[0]


def read_push_credentials(config: PlatformConfig) -> Tuple[str, str]:
    """Return (username, password) from Vault push path. Never logs values."""
    from cat_agent.platform.gateway import GatewayError, read_vault_kv_data

    path = config.registry_push_vault_path()
    try:
        data = read_vault_kv_data(config.vault_addr, path)
    except GatewayError as exc:
        raise BuildError(
            f'cannot read registry push credentials from Vault at {path}: {exc}'
        ) from exc
    user = str(data.get('username') or '').strip()
    password = str(data.get('password') or '').strip()
    if not user or not password:
        raise BuildError(
            f'Vault {path} missing username/password; run: cat-agent stack seed --registry'
        )
    return user, password


def _push_denied_message(config: PlatformConfig) -> str:
    return (
        'push denied — check the registry credentials in Vault at '
        f'{config.registry_push_vault_path()}'
    )


def _is_auth_failure(stderr: str) -> bool:
    low = (stderr or '').lower()
    return any(
        s in low
        for s in (
            'status 401',
            'status 403',
            '401 unauthorized',
            '403 forbidden',
            'unauthorized',
            'authentication required',
            'denied',
            'authorization failed',
        )
    )


def docker_login_and_push(
    config: PlatformConfig,
    local_tag: str,
    *,
    run: RunFn = _default_run,
    login_run: Optional[LoginFn] = None,
) -> str:
    """Tag *local_tag* for the remote registry, login with push creds, and push."""
    if config.is_local_registry():
        raise BuildError('refusing to push: registry is local')
    remote = config.image_ref(local_tag)
    user, password = read_push_credentials(config)
    host = registry_host(config)

    def _default_login() -> subprocess.CompletedProcess:
        return subprocess.run(
            ['docker', 'login', host, '-u', user, '--password-stdin'],
            input=password,
            capture_output=True,
            text=True,
            check=False,
        )

    login = (login_run or _default_login)()
    if login.returncode != 0:
        err = login.stderr or login.stdout or ''
        if _is_auth_failure(err):
            raise BuildError(_push_denied_message(config))
        raise BuildError(f'docker login failed for {host}: {err[:400]}')

    tag_r = run(['docker', 'tag', local_tag, remote], None)
    if tag_r.returncode != 0:
        raise BuildError(f'docker tag failed: {tag_r.stderr}')
    push_r = run(['docker', 'push', remote], None)
    if push_r.returncode != 0:
        err = push_r.stderr or push_r.stdout or ''
        if _is_auth_failure(err):
            raise BuildError(_push_denied_message(config))
        raise BuildError(f'docker push failed for {remote}: {err[:500]}')
    return remote


def _post_build_import_check(
    tag: str,
    entrypoint: str,
    *,
    remote: bool,
    base_image: str,
    run: RunFn,
) -> None:
    """Fail loudly if the image cannot import the entrypoint / framework."""
    if remote:
        # Remote: cat_agent must resolve from the base image (site-packages),
        # never from a shadowed COPY under /opt/cat-agent.
        script = (
            'import cat_agent, importlib, os, pathlib; '
            'p = pathlib.Path(cat_agent.__file__).resolve(); '
            'assert "/opt/cat-agent" not in p.as_posix(), ('
            '"cat_agent resolved to %s — source COPY shadowed the base image; '
            'remote builds must not COPY cat_agent" % p'
            '); '
            "importlib.import_module(os.environ['CAT_AGENT_ENTRYPOINT'].split(':')[0])"
        )
    else:
        script = (
            "import importlib, os; "
            "importlib.import_module(os.environ['CAT_AGENT_ENTRYPOINT'].split(':')[0])"
        )
    check = run(
        [
            'docker',
            'run',
            '--rm',
            '-e',
            f'CAT_AGENT_ENTRYPOINT={entrypoint}',
            '-e',
            'CAT_AGENT_MANAGED=1',
            '--entrypoint',
            'python',
            tag,
            '-c',
            script,
        ],
        None,
    )
    if check.returncode != 0:
        err = (check.stderr or check.stdout or '').strip()
        if remote and (
            'ModuleNotFoundError' in err
            or 'No module named' in err
            or 'cat_agent' in err.lower()
        ):
            raise BuildError(
                f'post-build import check failed for {tag}: base image '
                f'{base_image!r} does not provide an importable cat_agent '
                f'(remote mode does not COPY the source tree). '
                f'Run `cat-agent build-base` and set platform.base_image, '
                f'or use registry=local for monorepo iteration. Detail: {err[:600]}'
            )
        raise BuildError(f'post-build import check failed for {tag}: {err[:800]}')


def build_agent_image(
    manifest: AgentManifest,
    config: PlatformConfig,
    source_dir: Path,
    *,
    image_tag: Optional[str] = None,
    push: bool = False,
    run: RunFn = _default_run,
    login_run: Optional[LoginFn] = None,
) -> str:
    """Build image for *manifest*. Returns the local tag (not registry-prefixed)."""
    source_dir = source_dir.resolve()
    tag = image_tag or short_content_tag(manifest, source_dir, config=config)
    remote_mode = not config.is_local_registry()

    with tempfile.TemporaryDirectory(prefix='cat-agent-build-') as tmp:
        ctx = Path(tmp)
        stage_agent_build_context(manifest, config, source_dir, ctx)

        env_file = ctx / '.env'
        if env_file.exists():
            env_file.unlink()

        result = run(['docker', 'build', '-t', tag, '.'], ctx)
        if result.returncode != 0:
            raise BuildError(
                f'docker build failed for {tag}: {(result.stderr or result.stdout)[:1200]}'
            )

        _post_build_import_check(
            tag,
            manifest.runtime.entrypoint,
            remote=remote_mode,
            base_image=config.base_image,
            run=run,
        )

        if push and remote_mode:
            docker_login_and_push(config, tag, run=run, login_run=login_run)
        return tag


def build_base_image(
    config: PlatformConfig,
    dockerfile: Path,
    *,
    tag: str = 'cat-agent-runtime:latest',
    push: bool = False,
    run: RunFn = _default_run,
    login_run: Optional[LoginFn] = None,
) -> str:
    context = dockerfile.parent
    result = run(['docker', 'build', '-f', str(dockerfile), '-t', tag, '.'], context)
    if result.returncode != 0:
        raise BuildError(f'base image build failed: {(result.stderr or "")[:800]}')
    if push and not config.is_local_registry():
        docker_login_and_push(config, tag, run=run, login_run=login_run)
    return tag
