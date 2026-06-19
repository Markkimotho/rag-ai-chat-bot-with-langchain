import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("api client", () => {
  it("GET /api/models returns parsed body", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ supported: ["a"], installed: [] }));
    vi.stubGlobal("fetch", fetchMock);

    const models = await api.models();
    expect(models.supported).toEqual(["a"]);
    expect(fetchMock).toHaveBeenCalledWith("/api/models", expect.any(Object));
  });

  it("serializes JSON bodies for POST", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true, message: "x" }));
    vi.stubGlobal("fetch", fetchMock);

    await api.pullModel("mistral");
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ model: "mistral" });
  });

  it("throws ApiError with the server detail on non-2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "nope" }, 400)),
    );
    await expect(api.kbCount()).rejects.toMatchObject({
      name: "ApiError",
      status: 400,
      message: "nope",
    });
  });

  it("uploads files as multipart FormData", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ chunks_added: 1, detail: "ok" }));
    vi.stubGlobal("fetch", fetchMock);

    const file = new File(["%PDF"], "a.pdf", { type: "application/pdf" });
    await api.kbUpload([file]);
    const [, init] = fetchMock.mock.calls[0];
    expect(init.body).toBeInstanceOf(FormData);
  });

  it("ApiError is an Error subclass", () => {
    const e = new ApiError(500, "boom");
    expect(e).toBeInstanceOf(Error);
    expect(e.status).toBe(500);
  });
});
