export function shouldSendCollabUpdate({
  canEditWorkspace,
  collabReadOnly,
  hasUnsentEdit,
  ownsCurrentDocument,
  collabVersion,
}: {
  canEditWorkspace: boolean;
  collabReadOnly: boolean;
  hasUnsentEdit: boolean;
  ownsCurrentDocument: boolean;
  collabVersion: number;
}): boolean {
  return (
    canEditWorkspace && !collabReadOnly && hasUnsentEdit && ownsCurrentDocument && collabVersion > 0
  );
}
