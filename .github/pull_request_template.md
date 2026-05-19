## Summary

Describe what changed and why.

## Validation

- [ ] Backend quality checks and pytest pass (`docker build --target python-test -f docker/Dockerfile .` or CI)
- [ ] Frontend build passes when UI code changed (`docker build --target frontend-builder -f docker/Dockerfile .` or CI)
- [ ] Duplication check passes or intentional duplication is explained
- [ ] Relevant API/UI/runtime smoke checks completed

## Release Notes

- [ ] Database migrations are committed, if schema changed
- [ ] Configuration or environment variable changes are documented
- [ ] User-facing behavior changes are described