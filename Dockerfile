# syntax=docker/dockerfile:1
# Stage 1: Base image to install common dependencies and lock Python dependencies
FROM public.ecr.aws/docker/library/python:3.12-alpine3.20 AS python_packages

# Set terminal width (COLUMNS) and height (LINES)
ENV COLUMNS=300

# Define an argument to control whether to run uv lock (used for updating uv.lock)
ARG RUN_UV_LOCK=false

# Install common tools and dependencies (git is required for some Python packages)
RUN apk add --no-cache git

# Install uv from the official image (fast, single binary)
COPY --from=ghcr.io/astral-sh/uv:0.11.6@sha256:b1e699368d24c57cda93c338a57a8c5a119009ba809305cc8e86986d4a006754 /uv /uvx /usr/local/bin/

# Use a venv outside the project dir so docker-compose volume mounts don't hide it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Set the working directory inside the container
WORKDIR /usr/src/languagemodelcommon/

# Copy pyproject.toml and uv.lock to the working directory
COPY pyproject.toml uv.lock* /usr/src/languagemodelcommon/

# Conditionally run uv lock to update the uv.lock based on the argument provided
# If RUN_UV_LOCK is true, it regenerates the uv.lock file with the latest versions of dependencies
RUN if [ "$RUN_UV_LOCK" = "true" ]; then echo "Locking dependencies" && rm -f uv.lock && uv lock --verbose; fi

# Install all dependencies using the locked versions in uv.lock
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache \
    uv sync --frozen --all-extras --group dev --no-install-project --verbose

# Copy lock file for retrieval
RUN cp -f uv.lock /tmp/uv.lock

# Create necessary directories and list their contents (for debugging and verification)
RUN mkdir -p /opt/venv/lib/python3.12/site-packages && ls -halt /opt/venv/lib/python3.12/site-packages
RUN mkdir -p /opt/venv/bin && ls -halt /opt/venv/bin

# Check and print system and Python platform information (for debugging)
RUN python -c "import platform; print(platform.platform()); print(platform.architecture())"
RUN python -c "import sys; print(sys.platform, sys.version, sys.maxsize > 2**32)"

# Stage 2: Final runtime image for the application
FROM public.ecr.aws/docker/library/python:3.12-alpine3.20

# Set terminal width (COLUMNS) and height (LINES)
ENV COLUMNS=300

# Install common tools and dependencies (git is required for some Python packages)
RUN apk add --no-cache git

# Install uv for runtime use
COPY --from=ghcr.io/astral-sh/uv:0.11.6@sha256:b1e699368d24c57cda93c338a57a8c5a119009ba809305cc8e86986d4a006754 /uv /uvx /usr/local/bin/

# Set environment variables for project configuration
ENV PROJECT_DIR=/usr/src/languagemodelcommon
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

# Create the directory for Prometheus metrics
RUN mkdir -p ${PROMETHEUS_MULTIPROC_DIR}

# Set the working directory for the project
WORKDIR ${PROJECT_DIR}

# Copy the venv with all installed packages from the build stage
COPY --from=python_packages /opt/venv /opt/venv

# Copy pyproject.toml into the runtime image (uv.lock comes from Stage 1 below)
COPY pyproject.toml ${PROJECT_DIR}/

# Copy the application code into the runtime image
COPY ./languagemodelcommon ${PROJECT_DIR}

# Copy the uv.lock from the previous stage in case it was locked
COPY --from=python_packages /usr/src/languagemodelcommon/uv.lock ${PROJECT_DIR}/uv.lock
COPY --from=python_packages /tmp/uv.lock /tmp/uv.lock

# Create directories and list their contents (for debugging and verification)
RUN ls -halt /opt/venv/lib/python3.12/site-packages
RUN ls -halt /opt/venv/bin

# Expose port 5000 for the application
EXPOSE 5000

# Switch to the root user to perform user management tasks
USER root

# Create a restricted user (appuser) and group (appgroup) for running the application
RUN addgroup -S appgroup && adduser -S -h /etc/appuser appuser -G appgroup

# Ensure that the appuser owns the application files and directories
RUN chown -R appuser:appgroup ${PROJECT_DIR} /opt/venv ${PROMETHEUS_MULTIPROC_DIR}

# Switch to the restricted user to enhance security
USER appuser
