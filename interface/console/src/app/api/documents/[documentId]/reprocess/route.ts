import { NextRequest, NextResponse } from "next/server";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hms-client";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ documentId: string }> }
) {
  try {
    const { documentId } = await params;
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id");

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const response = await fetch(
      `${DATAPLANE_URL}/v1/default/banks/${bankId}/documents/${documentId}/reprocess`,
      {
        method: "POST",
        headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
      }
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error reprocessing document:", error);
    return NextResponse.json({ error: "Failed to reprocess document" }, { status: 500 });
  }
}
