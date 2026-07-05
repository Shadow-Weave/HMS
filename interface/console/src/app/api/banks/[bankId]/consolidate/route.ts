import { NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hms-client";

export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const response = await sdk.triggerConsolidation({
      client: lowLevelClient,
      path: { bank_id: bankId },
    });

    if (response.error) {
      console.error("API error triggering consolidation:", response.error);
      return NextResponse.json({ error: "Failed to trigger consolidation" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error triggering consolidation:", error);
    return NextResponse.json({ error: "Failed to trigger consolidation" }, { status: 500 });
  }
}
