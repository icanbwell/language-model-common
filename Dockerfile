# Stage 1: Base image to install common dependencies and lock Python dependencies
# This stage is responsible for setting up the environment and installing Python packages using Pipenv.
FROM public.ecr.aws/docker/library/python:3.12-alpine3.20 AS python_packages

# Set terminal width (COLUMNS) and height (LINES)
ENV COLUMNS=300

# Define an argument to control whether to run pipenv lock (used for updating Pipfile.lock)
ARG RUN_PIPENV_LOCK=false

# Install common tools and dependencies (git is required for some Python packages)
RUN apk add --no-cache git

# Install pipenv, a tool for managing Python project dependencies
RUN pip install pipenv

# Set the working directory inside the container
WORKDIR /usr/src/languagemodelcommon/

# Copy Pipfile and Pipfile.lock to the working directory
# Pipfile defines the Python packages required for the project
# Pipfile.lock ensures consistency by locking the exact versions of packages
COPY Pipfile* /usr/src/languagemodelcommon/

# Show the current pip configuration (for debugging purposes)
RUN pip config list

# Conditionally run pipenv lock to update the Pipfile.lock based on the argument provided
# If RUN_PIPENV_LOCK is true, it regenerates the Pipfile.lock file with the latest versions of dependencies
RUN if [ "$RUN_PIPENV_LOCK" = "true" ]; then echo "Locking Pipfile" && rm -f Pipfile.lock && pipenv lock --dev --clear --verbose --extra-pip-args="--prefer-binary"; fi

# Install all dependencies using the locked versions in Pipfile.lock
# --dev installs development dependencies, --system installs them globally in the container's Python environment
RUN pipenv sync --dev --system --verbose --extra-pip-args="--prefer-binary"

# Create necessary directories and list their contents (for debugging and verification)
RUN mkdir -p /usr/local/lib/python3.12/site-packages && ls -halt /usr/local/lib/python3.12/site-packages
RUN mkdir -p /usr/local/bin && ls -halt /usr/local/bin

# Check and print system and Python platform information (for debugging)
RUN python -c "import platform; print(platform.platform()); print(platform.architecture())"
RUN python -c "import sys; print(sys.platform, sys.version, sys.maxsize > 2**32)"

# Debug pip installation and list installed packages with verbosity
RUN pip debug --verbose
RUN pip list -v

# Stage 2: Final runtime image for the application
# This stage creates a minimal image with only the runtime dependencies and the application code.
FROM public.ecr.aws/docker/library/python:3.12-alpine3.20

# Set terminal width (COLUMNS) and height (LINES)
ENV COLUMNS=300

# Define an argument to control whether to run pipenv lock (not used in this stage)
ARG RUN_PIPENV_LOCK=false

# Install common tools and dependencies (git is required for some Python packages)
RUN apk add --no-cache git
# Install runtime dependencies required by the application (e.g., for shapely, grpcio, scipy, google-crc32 and numpy)
# You can use auditwheel to check any package and identify the native library dependencies
# RUN apk add --no-cache curl libstdc++ libffi

# Install pipenv to manage and run the application
RUN pip install --no-cache-dir pipenv

# Set environment variables for project configuration
ENV PROJECT_DIR=/usr/src/languagemodelcommon
ENV FLASK_APP=languagemodelcommon.api
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

# Create the directory for Prometheus metrics
RUN mkdir -p ${PROMETHEUS_MULTIPROC_DIR}

# Set the working directory for the project
WORKDIR ${PROJECT_DIR}

# Copy the Pipfile and Pipfile.lock files into the runtime image
COPY Pipfile* ${PROJECT_DIR}

# Copy installed Python packages from the previous stage
COPY --from=python_packages /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python_packages /usr/local/bin /usr/local/bin

# Copy the application code into the runtime image
COPY ./languagemodelcommon ${PROJECT_DIR}

# Copy the Pipfile.lock from the previous stage in case it was locked
COPY --from=python_packages /usr/src/languagemodelcommon/Pipfile.lock /usr/src/languagemodelcommon/Pipfile.lock

# Copy Pipfile.lock to a temporary directory so it can be retrieved if needed
COPY --from=python_packages /usr/src/languagemodelcommon/Pipfile.lock /tmp/Pipfile.lock

# Create directories and list their contents (for debugging and verification)
RUN mkdir -p /usr/local/lib/python3.12/site-packages && ls -halt /usr/local/lib/python3.12/site-packages
RUN mkdir -p /usr/local/bin && ls -halt /usr/local/bin

# Install the dependencies using pipenv in the final runtime environment
RUN pipenv sync --dev --system --extra-pip-args="--prefer-binary"

# Expose port 5000 for the application
EXPOSE 5000

# Switch to the root user to perform user management tasks
USER root

# Create a restricted user (appuser) and group (appgroup) for running the application
RUN addgroup -S appgroup && adduser -S -h /etc/appuser appuser -G appgroup

# Ensure that the appuser owns the application files and directories
RUN chown -R appuser:appgroup ${PROJECT_DIR} /usr/local/lib/python3.12/site-packages /usr/local/bin ${PROMETHEUS_MULTIPROC_DIR}

# Switch to the restricted user to enhance security
USER appuser
