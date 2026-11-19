from engine.llm.parse import loads_llm_json


def test_loose_reply_json() -> None:
    raw = """{
  items: [
    {
      id: "c1",
      suggested_reply: "Thanks!",
      notes: [],
    },
  ]
}"""
    data = loads_llm_json(raw)
    assert data["items"][0]["suggested_reply"] == "Thanks!"