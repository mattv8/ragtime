# MCP token refresh overlap

## Scope

This server behavior applies only to interactive MCP OAuth refresh-token
renewal. Dashboard bridge authentication, client-credentials grants, and
password grants are separate authentication lanes and do not receive this
overlap behavior.

## Renewal behavior

The server retains strict refresh-token rotation by default. For the MCP
interactive lane only, it allows an exact duplicate use of a just-consumed
refresh token for a fixed 10-second interval measured from the original
consumption. The interval never slides or extends.

For an accepted duplicate, the server returns the same deterministically
derived successor refresh token as the original request. It stores only token
hashes; it does not persist raw predecessor or successor tokens. The server
mints an access token for each accepted response, so a duplicate response can
have a later issuance time. Its lifetime uses the unchanged configured
access-token duration and is capped by the same finite authorization-grant
expiry; neither request extends the grant.

The duplicate must match the same live, unconsumed successor and all existing
grant, user, MFA, client, resource, scope, security-generation, expiry, and
revocation checks.

## Strict replay handling

The server returns `invalid_grant` and revokes the refresh-token family when
a replay is outside the 10-second interval, has no matching successor, or the
successor has already been consumed. A revoked authorization cannot be renewed;
the user must complete a fresh login.

## Client guidance

Clients should single-flight refresh requests for one authorization. Clients
running in multiple processes should also coordinate through a shared,
atomic authentication store so only one process performs a rotation and the
others consume the saved result. This server-side tolerance does not mean the
client's refresh coordination has been implemented or fixed.

## Trade-offs and rollout limits

The overlap is a bounded relaxation of replay detection: a party holding a
predecessor refresh token within the 10-second interval can obtain the same
successor token. It does not guarantee that a later rotation detects that
theft. Finite grants and strict replay handling outside the interval remain in
effect.

Slow retries can arrive after the interval. Mixed server versions can create
successors that the new duplicate path cannot recover, and signing-key changes
can invalidate deterministic successor derivation. These cases may require a
fresh login.
