import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { JobsTableFrame } from './JobsTableFrame';

describe('JobsTableFrame', () => {
  it('keeps the shared scroll wrapper and table contracts with extensions', () => {
    const { container } = render(
      <JobsTableFrame
        id="directory"
        wrapperClassName="directory-wrap"
        tableClassName="directory-table"
      >
        <tbody>
          <tr>
            <td>Row</td>
          </tr>
        </tbody>
      </JobsTableFrame>,
    );
    expect(
      container.querySelector('#directory-frame.jobs-table-wrapper.directory-wrap'),
    ).toBeTruthy();
    expect(container.querySelector('table#directory.jobs-table.directory-table')).toBeTruthy();
  });
});
