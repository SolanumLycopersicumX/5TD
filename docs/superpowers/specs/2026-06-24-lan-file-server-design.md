# LAN File Server Design

## Goal

Provide a local file-transfer server for devices on the same LAN. The server lets the local machine and LAN clients exchange files through one isolated shared directory.

## Scope

- Shared directory: `lan_share/` under the repository root.
- Server script: `tools/lan_file_server.py`.
- Network binding: `0.0.0.0` so other LAN devices can connect.
- Default port: `8000`; if unavailable, the launcher chooses another free port.
- Supported operations: browse files, download files, upload files, and create subdirectories implicitly through normal file placement only when handled safely.
- No third-party Python dependency.

## Safety Boundaries

- The server exposes only `lan_share/`, not the whole repository.
- Request paths are normalized and rejected if they escape the shared directory.
- Uploaded filenames are reduced to safe path components.
- Existing uploaded files are not overwritten silently; the server adds a numeric suffix.
- The service is meant for trusted LAN use, not the public internet.

## User Flow

1. Start the script from the repository.
2. The script ensures `lan_share/` exists.
3. The terminal prints local and LAN URLs.
4. A browser opens the URL from any LAN device.
5. Users download files by clicking links and upload files with a form.

## Testing

Focused tests cover:

- Path containment rejects traversal.
- Upload filenames are sanitized.
- Repeated uploads produce unique filenames.
- The HTTP handler can save a multipart upload into the shared directory.
