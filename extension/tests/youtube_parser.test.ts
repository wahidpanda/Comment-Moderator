import { readFileSync } from "fs";
import { resolve } from "path";
import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";
import { parseCommentsFromDocument } from "../src/parsers/youtube";

describe("YouTube parser", () => {
  it("parses fixture HTML", () => {
    const html = readFileSync(resolve(__dirname, "../fixtures/youtube_comments.html"), "utf-8");
    const doc = new JSDOM(html).window.document;
    const comments = parseCommentsFromDocument(doc);
    expect(comments).toHaveLength(2);
    expect(comments[0].text).toContain("thank you");
  });
});