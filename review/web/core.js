/* Shared validation. Also loaded directly by the dependency-free Node tests. */
(function (root) {
  'use strict';
  const criteria = ['accuracy', 'usefulness', 'non_redundancy', 'clarity_timing'];
  const collisions = ['dialogue_overlap', 'narration_overlap'];
  const preferences = ['X', 'Y', 'Tie', 'Cannot assess'];
  const collisionValues = ['Yes', 'No', 'Uncertain'];
  function mediaPath(value) {
    if (typeof value !== 'string' || !value || /[\\\x00-\x1f]/.test(value)) return false;
    if (/^https?:\/\//i.test(value)) return true;
    return !value.startsWith('/') && !value.includes(':') && !value.split('/').includes('..');
  }
  function validatePackage(p) {
    if (!p || p.schema_version !== 1 || typeof p.study_id !== 'string' || !p.study_id ||
        typeof p.package_id !== 'string' || !p.package_id || typeof p.reviewer !== 'string' || !p.reviewer ||
        typeof p.demo !== 'boolean' || !Array.isArray(p.items) || !p.items.length) throw Error('Invalid review package. Use the package builder.');
    const ids = new Set();
    for (const item of p.items) {
      if (!item || typeof item.id !== 'string' || !item.id || ids.has(item.id)) throw Error('Missing or duplicate comparison ID.');
      ids.add(item.id);
      if (!item.media || !['original','X','Y'].every(k => mediaPath(item.media[k]))) throw Error('Invalid media path. Use relative paths or HTTP(S) URLs.');
    }
    return p;
  }
  function validateRating(r) {
    if (!r || !preferences.includes(r.preference)) return 'Choose X, Y, Tie, or Cannot assess.';
    if (typeof r.evidence !== 'string' || !r.evidence.trim()) return 'Explain your preference or why you cannot assess the item.';
    if (!r.independent) return 'Confirm that you reviewed the evidence independently.';
    if (r.preference !== 'Cannot assess') {
      for (const v of ['X','Y']) {
        for (const c of criteria) if (!Number.isInteger(r[v]?.[c]) || r[v][c] < 1 || r[v][c] > 5) return `Rate every criterion for version ${v}.`;
        for (const c of collisions) if (!collisionValues.includes(r[v]?.[c])) return `Complete both overlap checks for version ${v}.`;
      }
    }
    return '';
  }
  function validateImport(data, p) {
    if (!data || data.schema_version !== 1 || data.study_id !== p.study_id || data.package_id !== p.package_id || data.reviewer !== p.reviewer || data.demo !== p.demo || !Array.isArray(data.ratings)) throw Error('This export belongs to a different package, study, or reviewer.');
    const ids = new Set(p.items.map(i=>i.id)), seen = new Set();
    for (const r of data.ratings) {
      if (!r || !ids.has(r.item_id) || seen.has(r.item_id) || typeof r.completed !== 'boolean') throw Error('Invalid or duplicate review record.');
      seen.add(r.item_id);
      if (r.completed && validateRating(r)) throw Error('The export contains an incomplete review marked complete.');
    }
    return data.ratings;
  }
  const api = {criteria, collisions, preferences, validatePackage, validateRating, validateImport};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ReviewCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : window);
