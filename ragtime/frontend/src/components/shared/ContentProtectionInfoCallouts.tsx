import { InfoCallout } from './InfoCallout';

export function CategoriesInfoCallout() {
  return (
    <InfoCallout id="content-protection-categories-info">
      <p>
        Categories define how user messages and assistant responses are classified. Access levels
        determine who can access information in each category.
      </p>
      <p>
        Content that does not match a restricted category can pass when classification succeeds. A
        matched category at or above the applicable threshold denies access for an audience without
        a grant for that category.
      </p>
    </InfoCallout>
  );
}

export function AccessLevelsInfoCallout() {
  return (
    <InfoCallout id="content-protection-access-levels-info">
      <p>
        Access levels control which categories users may access. Assign levels to groups; grants
        from mapped levels combine. Users without a mapped group receive the default level.
      </p>
      <p>
        When classification runs, bypass attempts are checked too. Skipping classification disables
        these checks.
      </p>
    </InfoCallout>
  );
}
