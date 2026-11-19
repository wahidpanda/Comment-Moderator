import { describe, expect, it } from "vitest";
import { extractYouTubeVideoId } from "./youtube";

describe("extractYouTubeVideoId", () => {
  it("parses standard watch URLs", () => {
    expect(extractYouTubeVideoId("https://www.youtube.com/watch?v=IGEGgAKtA-4")).toBe("IGEGgAKtA-4");
  });

  it("parses watch URLs when v is not the first query param", () => {
    expect(
      extractYouTubeVideoId("https://www.youtube.com/watch?si=xYz&v=IGEGgAKtA-4&feature=share")
    ).toBe("IGEGgAKtA-4");
  });

  it("parses bare ids and shorts", () => {
    expect(extractYouTubeVideoId("IGEGgAKtA-4")).toBe("IGEGgAKtA-4");
    expect(extractYouTubeVideoId("https://www.youtube.com/shorts/IGEGgAKtA-4")).toBe("IGEGgAKtA-4");
  });
});