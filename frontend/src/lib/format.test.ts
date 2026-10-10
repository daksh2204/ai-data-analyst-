import { describe, expect, it } from "vitest";
import { compact, formatNumber, isNumericColumn, percent, pretty, toCsv } from "./format";
import { parseSseChunk } from "../api/client";

describe("number formatting", () => {
  it("compacts large values", () => {
    expect(compact(3_025_469)).toBe("3.03M");
    expect(compact(12_500)).toBe("12.5K");
    expect(compact(1_200_000_000)).toBe("1.20B");
    expect(compact(512.4)).toBe("512");
  });
  it("formats table cells", () => {
    expect(formatNumber(1234567)).toBe("1,234,567");
    expect(formatNumber(1.23456)).toBe("1.23");
    expect(formatNumber(null)).toBe("");
    expect(formatNumber(Number.NaN)).toBe("");
  });
  it("signs percentages", () => {
    expect(percent(0.286)).toBe("+28.6%");
    expect(percent(-0.05)).toBe("-5.0%");
  });
  it("prettifies identifiers", () => {
    expect(pretty("sales_channel")).toBe("Sales channel");
  });
});

describe("tables", () => {
  it("detects numeric columns", () => {
    expect(isNumericColumn([[1, "a"], [null, "b"], [3, "c"]], 0)).toBe(true);
    expect(isNumericColumn([[1, "a"]], 1)).toBe(false);
  });
  it("escapes CSV", () => {
    expect(toCsv(["a", "b"], [["x,y", 'say "hi"']])).toBe('a,b\n"x,y","say ""hi"""');
  });
});

describe("SSE parsing", () => {
  it("splits complete events and keeps the remainder", () => {
    const { events, rest } = parseSseChunk('data: {"type":"start"}\n\ndata: {"type":"token","text":"Hi"}\n\ndata: {"ty');
    expect(events).toEqual(['{"type":"start"}', '{"type":"token","text":"Hi"}']);
    expect(rest).toBe('data: {"ty');
  });
  it("handles CRLF and multi-line data", () => {
    const { events } = parseSseChunk("data: a\r\ndata: b\r\n\r\n");
    expect(events).toEqual(["a\nb"]);
  });
});
