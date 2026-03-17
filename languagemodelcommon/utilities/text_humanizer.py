class Humanizer:
    @staticmethod
    def humanize_tool_name(key: str) -> str:
        normalized = key.replace("-", "_")
        parts = [part for part in normalized.split("_") if part]
        if not parts:
            return key
        uppercase_tokens = {"id", "ids", "url", "uri", "oidc", "jwt", "mcp", "fhir"}
        humanized_parts = [
            part.upper() if part.lower() in uppercase_tokens else part.capitalize()
            for part in parts
        ]
        return " ".join(humanized_parts)
