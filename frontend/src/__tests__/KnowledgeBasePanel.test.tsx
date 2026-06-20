import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

const apiMock = vi.hoisted(() => ({
  kbCount: vi.fn(),
  kbSources: vi.fn(),
  kbUpload: vi.fn(),
  kbScrape: vi.fn(),
  kbClear: vi.fn(),
}));
vi.mock("../api/client", () => ({ api: apiMock, ApiError: class extends Error {} }));

import { KnowledgeBaseProvider } from "../context/KnowledgeBaseContext";
import { KnowledgeBasePanel } from "../components/KnowledgeBasePanel";

function renderPanel() {
  return render(
    <KnowledgeBaseProvider>
      <KnowledgeBasePanel onClose={() => {}} />
    </KnowledgeBaseProvider>,
  );
}

afterEach(() => {
  Object.values(apiMock).forEach((m) => m.mockReset());
});

describe("KnowledgeBasePanel", () => {
  it("lists indexed files with chunk counts", async () => {
    apiMock.kbCount.mockResolvedValue({ count: 52 });
    apiMock.kbSources.mockResolvedValue({
      sources: [
        { source: "algo.pdf", chunks: 12, type: "pdf" },
        { source: "wikipedia.org", chunks: 40, type: "web" },
      ],
    });
    renderPanel();
    expect(await screen.findByText("algo.pdf")).toBeInTheDocument();
    expect(screen.getByText("wikipedia.org")).toBeInTheDocument();
    expect(screen.getByText("2 files · 52 chunks")).toBeInTheDocument();
  });

  it("shows an empty state when nothing is indexed", async () => {
    apiMock.kbCount.mockResolvedValue({ count: 0 });
    apiMock.kbSources.mockResolvedValue({ sources: [] });
    renderPanel();
    expect(await screen.findByText(/No files yet/)).toBeInTheDocument();
  });

  it("acknowledges a successful upload and refreshes the list", async () => {
    apiMock.kbCount
      .mockResolvedValueOnce({ count: 0 })
      .mockResolvedValue({ count: 3 });
    apiMock.kbSources
      .mockResolvedValueOnce({ sources: [] })
      .mockResolvedValue({ sources: [{ source: "notes.pdf", chunks: 3, type: "pdf" }] });
    apiMock.kbUpload.mockResolvedValue({ chunks_added: 3, detail: "notes.pdf: 3 chunks" });

    const { container } = renderPanel();
    await screen.findByText(/No files yet/);

    const fileInput = container.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    const file = new File(["%PDF"], "notes.pdf", { type: "application/pdf" });
    await userEvent.upload(fileInput, file);

    expect(await screen.findByText(/notes\.pdf: 3 chunks/)).toBeInTheDocument();
    expect(await screen.findByText("notes.pdf")).toBeInTheDocument();
    expect(apiMock.kbUpload).toHaveBeenCalledOnce();
  });

  it("scrapes a topic and shows acknowledgement", async () => {
    apiMock.kbCount.mockResolvedValue({ count: 0 });
    apiMock.kbSources.mockResolvedValue({ sources: [] });
    apiMock.kbScrape.mockResolvedValue({ chunks_added: 80, detail: "Added 80 chunks from web." });

    renderPanel();
    await screen.findByText(/No files yet/);

    await userEvent.type(
      screen.getByLabelText("Scrape a web topic"),
      "binary search",
    );
    await userEvent.click(screen.getByRole("button", { name: "Index" }));

    await waitFor(() =>
      expect(apiMock.kbScrape).toHaveBeenCalledWith("binary search", 3),
    );
    expect(await screen.findByText(/Added 80 chunks from web/)).toBeInTheDocument();
  });
});
