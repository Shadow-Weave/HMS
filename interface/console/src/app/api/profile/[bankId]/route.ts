import { NextRequest, NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hms-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;
    const response = await sdk.getBankProfile({
      client: lowLevelClient,
      path: { bank_id: bankId },
    });
    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error fetching bank profile:", error);
    return NextResponse.json({ error: "Failed to fetch bank profile" }, { status: 500 });
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;
    const body = await request.json();

    const response = await sdk.createOrUpdateBank({
      client: lowLevelClient,
      path: { bank_id: bankId },
      body: body,
    });
    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error updating bank profile:", error);
    return NextResponse.json({ error: "Failed to update bank profile" }, { status: 500 });
  }
}
