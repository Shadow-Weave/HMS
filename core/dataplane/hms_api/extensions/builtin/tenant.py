"""Built-in tenant extension implementations."""

from hms_api.config import get_config
from hms_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hms_api.models import RequestContext


class DefaultTenantExtension(TenantExtension):
    """
    Default single-tenant extension with no authentication.

    This is the default extension used when no tenant extension is configured.
    It provides single-tenant behavior using the configured schema from
    HMS_API_DATABASE_SCHEMA (defaults to 'public').

    Features:
    - No authentication required (passes all requests)
    - Uses configured schema from environment
    - Perfect for single-tenant deployments without auth

    Configuration:
        HMS_API_DATABASE_SCHEMA=your-schema (optional, defaults to 'public')

    This is automatically enabled by default. To use custom authentication,
    configure a different tenant extension:
        HMS_API_TENANT_EXTENSION=hms_api.extensions.builtin.tenant:ApiKeyTenantExtension
    """

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        # Cache the schema at initialization for consistency
        # Support explicit schema override via config, otherwise use environment
        self._schema = config.get("schema", get_config().database_schema)

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """Return configured schema without any authentication."""
        return TenantContext(schema_name=self._schema)

    async def list_tenants(self) -> list[Tenant]:
        """Return configured schema for single-tenant setup."""
        return [Tenant(schema=self._schema)]


class ApiKeyTenantExtension(TenantExtension):
    """
    Built-in tenant extension that validates API key against an environment variable.

    This is a simple implementation that:
    1. Validates the API key matches HMS_API_TENANT_API_KEY
    2. Returns the configured schema (HMS_API_DATABASE_SCHEMA, default 'public')
       for all authenticated requests

    Configuration:
        HMS_API_TENANT_EXTENSION=hms_api.extensions.builtin.tenant:ApiKeyTenantExtension
        HMS_API_TENANT_API_KEY=your-secret-key
        HMS_API_DATABASE_SCHEMA=your-schema (optional, defaults to 'public')
        HMS_API_TENANT_MCP_AUTH_DISABLED=true (optional, disable auth for MCP endpoints)

    For multi-tenant setups with separate schemas per tenant, implement a custom
    TenantExtension that looks up the schema based on the API key or token claims.
    """

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self.expected_api_key = config.get("api_key")
        if not self.expected_api_key:
            raise ValueError("HMS_API_TENANT_API_KEY is required when using ApiKeyTenantExtension")
        # Allow disabling MCP auth for backwards compatibility
        self.mcp_auth_disabled = config.get("mcp_auth_disabled", "").lower() in ("true", "1", "yes")

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """Validate API key and return configured schema context."""
        if context.api_key != self.expected_api_key:
            raise AuthenticationError("Invalid API key")
        return TenantContext(schema_name=get_config().database_schema)

    async def list_tenants(self) -> list[Tenant]:
        """Return configured schema for single-tenant setup."""
        return [Tenant(schema=get_config().database_schema)]

    async def authenticate_mcp(self, context: RequestContext) -> TenantContext:
        """
        Authenticate MCP requests.

        If mcp_auth_disabled is set, skip authentication for backwards compatibility.
        Otherwise, delegate to authenticate().
        """
        if self.mcp_auth_disabled:
            return TenantContext(schema_name=get_config().database_schema)
        return await self.authenticate(context)
