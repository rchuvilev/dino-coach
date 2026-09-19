/**
 * Persistent, exportable run log.
 *
 * The old log was six lines of `textContent` that vanished on reload and was
 * invisible to Export. With a trained model in the loop, the log is the only
 * record of WHAT the weights were corrected on - so it now persists, clears
 * with Reset, and travels inside the export file alongside the model.
 *
 * Entries are structured rather than pre-formatted strings: the same record
 * can then be rendered for the panel, filtered by kind, and re-read after an
 * import without parsing text back apart.
 */

const CAP = 400;   // bounded like every other store in this project

export class LogStore {
  constructor(cap = CAP) {
    this.cap = cap;
    this.entries = [];
    this.seq = 0;
  }

  /**
   * @param {string} kind  episode | train | generation | model | system | error
   * @param {string} msg   human-readable summary
   * @param {object} [data] structured detail, kept for export and filtering
   */
  add(kind, msg, data) {
    this.entries.push({
      i: ++this.seq,
      t: Date.now(),
      kind,
      msg,
      ...(data ? { data } : {}),
    });
    if (this.entries.length > this.cap) this.entries.shift();
  }

  /** Newest first, optionally filtered, for the on-screen panel. */
  recent(n = 6, kinds = null) {
    const list = kinds
      ? this.entries.filter((e) => kinds.includes(e.kind))
      : this.entries;
    return list.slice(-n).reverse();
  }

  /** One line per entry, newest first. */
  render(n = 6, kinds = null) {
    return this.recent(n, kinds)
      .map((e) => `${e.msg}`)
      .join("\n");
  }

  counts() {
    const out = {};
    for (const e of this.entries) out[e.kind] = (out[e.kind] || 0) + 1;
    return out;
  }

  toJSON() {
    return { cap: this.cap, seq: this.seq, entries: this.entries };
  }

  static fromJSON(o) {
    const s = new LogStore(o && o.cap ? o.cap : CAP);
    if (o && Array.isArray(o.entries)) {
      s.entries = o.entries.slice(-s.cap);
      s.seq = o.seq || s.entries.length;
    }
    return s;
  }

  clear() {
    this.entries = [];
    this.seq = 0;
  }
}
