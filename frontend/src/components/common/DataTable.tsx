import { useMemo, useState } from "react";
import { Download } from "lucide-react";
import type { TablePayload } from "../../types";
import { download, formatNumber, isNumericColumn, pretty, toCsv } from "../../lib/format";

interface Props {
  table: TablePayload;
  title?: string;
  initialRows?: number;
  dense?: boolean;
  filename?: string;
  rawHeaders?: boolean;
}

export function DataTable({ table, title, initialRows = 12, dense = false, filename, rawHeaders = false }: Props) {
  const [expanded, setExpanded] = useState(false);
  const numeric = useMemo(() => table.columns.map((_, i) => isNumericColumn(table.rows, i)), [table]);
  const rows = expanded ? table.rows : table.rows.slice(0, initialRows);
  const hidden = table.rows.length - rows.length;

  return (
    <div className={`data-table ${dense ? "is-dense" : ""}`}>
      <div className="data-table__scroll" tabIndex={0} aria-label={title ?? "Result table"}>
        <table>
          <thead>
            <tr>
              {table.columns.map((c, i) => (
                <th key={c + i} className={numeric[i] ? "num" : undefined} scope="col">
                  {rawHeaders ? c : pretty(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri}>
                {r.map((v, ci) => (
                  <td key={ci} className={numeric[ci] ? "num" : undefined}>
                    {v === null || v === undefined ? <span className="null">—</span> : formatNumber(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="data-table__foot">
        <span>
          {table.row_count.toLocaleString("en-GB")} row{table.row_count === 1 ? "" : "s"}
          {table.truncated ? " (truncated)" : ""}
        </span>
        <span className="data-table__actions">
          {hidden > 0 && (
            <button type="button" className="link-button" onClick={() => setExpanded(true)}>
              Show {Math.min(hidden, 200)} more
            </button>
          )}
          {expanded && table.rows.length > initialRows && (
            <button type="button" className="link-button" onClick={() => setExpanded(false)}>
              Show fewer
            </button>
          )}
          <button
            type="button"
            className="icon-button"
            title="Download CSV"
            aria-label="Download as CSV"
            onClick={() => download(`${filename ?? "result"}.csv`, toCsv(table.columns, table.rows))}
          >
            <Download size={15} aria-hidden />
          </button>
        </span>
      </div>
    </div>
  );
}
