# syntax=docker/dockerfile:1
FROM public.ecr.aws/docker/library/python:3.12-alpine3.20

# Install git, build-essential, and uv
RUN apk add --no-cache git build-base
COPY --from=ghcr.io/astral-sh/uv:0.11.16@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv (locked versions, skip building project itself)
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache \
    uv sync --frozen --all-extras --group dev --no-install-project --verbose

# Set the working directory
WORKDIR /sourcecode

# Non-root user for running pre-commit. UID/GID are build args so they can be
# set to match the host caller (see pre-commit-hook) -- /sourcecode and the
# git common dir are bind-mounted from the host, so the in-container user
# needs the same UID/GID to read/write them without permission errors.
ARG USER_UID=1000
ARG USER_GID=1000
RUN (getent group "$USER_GID" || addgroup -g "$USER_GID" appgroup) && \
    adduser -D -u "$USER_UID" -G "$(getent group "$USER_GID" | cut -d: -f1)" -h /home/appuser appuser && \
    mkdir -p /home/appuser/.cache/pre-commit && \
    chown -R "$USER_UID:$USER_GID" /home/appuser /opt/venv /sourcecode

# Use --global, not --system: `git config` writes its target via a
# temp-file-then-rename *in the same directory* as the target file, so it
# needs write access on that directory, not just the file. /etc is root-owned
# (0755) regardless of who owns /etc/gitconfig itself, so appuser can never
# write there -- this is what broke CI (`could not lock config file
# /etc/gitconfig: Permission denied`) despite the file being chowned to
# appuser. /home/appuser is chowned to appuser above, so --global's target
# directory is writable by appuser both now (build time, as root, via the
# explicit $HOME below) and later (runtime, as appuser, for the dynamic
# $GIT_COMMON_DIR safe.directory add in pre-commit-hook).
ENV HOME=/home/appuser
RUN git config --global --add safe.directory /sourcecode && \
    chown "$USER_UID:$USER_GID" /home/appuser/.gitconfig

USER appuser

CMD ["pre-commit", "run", "--all-files"]
