# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Herbert, please report it
responsibly by emailing **n0entry.social@gmail.com** with the subject line
"[SECURITY]" and a detailed description of the issue.

**Please do not open a public GitHub issue for security vulnerabilities.**

Include the following information in your report:

- A clear description of the vulnerability
- Steps to reproduce (if applicable)
- Affected components or versions
- Potential impact of the vulnerability
- Your suggested fix (if you have one)

## What to Expect

- You will receive an acknowledgment of your report within 48 hours
- We will investigate the vulnerability and provide a timeline for a fix
- We will work with you to understand the issue and develop a patch
- Once a fix is ready, we will release it and coordinate disclosure timing with you
- You will be credited in the security advisory (unless you prefer anonymity)

## Scope

This security policy applies to vulnerabilities in:

- All source code in the `/mod`, `/nn`, and `/rl` components
- All dependencies declared in each component's build configuration
- The `/bot` component (if you have access to it)

Out of scope:

- Security issues in Minecraft itself, Hypixel's servers, or third-party services
  (Discord, Mineflayer, etc.) — please report those to the respective projects
- Social engineering or user credential misuse
- Issues requiring physical access to systems

## Security Considerations

### Data Privacy

- The `/mod` component logs gameplay data that may contain player information.
  By default, player usernames are hashed using SHA-256 rather than logged in
  clear text. Players can opt in to include their raw username in session data.
- Session data is uploaded only to a Discord webhook URL configured by the player.
- The `/bot` component validates session data before storing it. See its README
  for details on data handling (if you have access to it).

### Passive Logging Only

- The `/mod` component is purely observational: it never injects input, sends
  packets, or automates gameplay.
- The `/nn` component is an offline training pipeline with no network access
  to game servers.
- The `/rl` component automates bots only against a private, self-hosted
  Minecraft server under developer control — never against Hypixel or other
  third-party servers.

### Secrets Management

- No API keys, tokens, Discord webhook URLs, or other secrets should ever be
  hardcoded in source code.
- All secrets are provided via environment variables at runtime.
- The `.gitignore` is configured to exclude `.env` files and prevent accidental
  secret commits.

## Previous Vulnerabilities

None reported as of 2025-08-05. As the project matures and sees wider use,
this section will be updated with any disclosed vulnerabilities and their
resolution.
