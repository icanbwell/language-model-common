"""GraphQL authentication utilities for verifying request tokens."""

from fastapi import HTTPException, Request, status
from oidcauthlib.auth.models.token import Token
from oidcauthlib.auth.token_reader import TokenReader
from oidcauthlib.container.container_registry import ContainerRegistry
from oidcauthlib.container.interfaces import IContainer

from languagemodelcommon.utilities.logger.log_levels import logger


async def verify_graphql_token(request: Request) -> Token:
    """Verify authentication token for GraphQL requests.

    Extracts and validates the Bearer token from the Authorization header
    using the configured TokenReader from the DI container.

    Args:
        request: The incoming FastAPI request

    Returns:
        The validated Token object with claims

    Raises:
        HTTPException: 401 Unauthorized if authentication fails
    """
    container: IContainer = ContainerRegistry.get_current()
    token_reader: TokenReader = container.resolve(TokenReader)

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )

    try:
        token_str = token_reader.extract_token(authorization_header=auth_header)
        if not token_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format",
            )
        # Use async token verification
        token_item: Token | None = await token_reader.verify_token_async(
            token=token_str
        )
        if not token_item:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        return token_item
    except HTTPException:
        raise
    except Exception:
        logger.warning("Authentication failed for GraphQL request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
