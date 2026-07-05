# Privacy Policy

The HMS Dify plugin sends data to a HMS API instance (HMS Cloud or self-hosted) configured by the workflow author.

## Data sent

- **Retain**: the content you pass to the tool, plus any tags, are sent to HMS to be stored as memories. HMS extracts facts from the content using an LLM.
- **Recall** / **Reflect**: the query string is sent to HMS; relevant memories from the configured bank are returned.

## What is stored

- All retained content, extracted facts, and metadata are stored in the HMS instance you configure.
- The plugin itself does not retain any data — it is a thin adapter to HMS's API.

## Where data goes

- **HMS Cloud (`https://api.hms.local`)**: data is processed and stored by HMS, Inc. See [https://docs.hms.local/privacy](https://docs.hms.local/privacy).
- **Self-hosted HMS**: data stays inside your own infrastructure.

## Credentials

The API key you configure is stored as a Dify secret credential and is sent to HMS on each tool invocation in the `Authorization: Bearer` header.
