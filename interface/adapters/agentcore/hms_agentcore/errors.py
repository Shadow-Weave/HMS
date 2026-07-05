class HMSAgentCoreError(Exception):
    """Exception raised when a HMS memory operation fails inside AgentCore Runtime."""

    pass


class BankResolutionError(HMSAgentCoreError):
    """Raised when bank ID resolution fails — fails closed to prevent memory leakage."""

    pass
