import { NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hms-client";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;
    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const { searchParams } = new URL(request.url);
    const q = searchParams.get("q");
    const source = searchParams.get("source");
    const limit = searchParams.get("limit");
    const offset = searchParams.get("offset");

    const queryParams = new URLSearchParams();
    if (q) queryParams.append("q", q);
    if (source) queryParams.append("source", source);
    if (limit) queryParams.append("limit", limit);
    if (offset) queryParams.append("offset", offset);

    const url = dataplaneBankUrl(bankId, `/tags${queryParams.toString() ? `?${queryParams}` : ""}`);
    const response = await fetch(url, { method: "GET", headers: getDataplaneHeaders() });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error listing tags:", errorText);
      return NextResponse.json({ error: "Failed to list tags" }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error listing tags:", error);
    return NextResponse.json({ error: "Failed to list tags" }, { status: 500 });
  }
}
