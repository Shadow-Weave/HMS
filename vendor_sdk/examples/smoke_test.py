from hms_vendor_sdk import HMSVendorClient, SessionRecord


def main() -> None:
    client = HMSVendorClient.from_env()

    sessions = [
        SessionRecord(
            session_id="demo-session-1",
            timestamp="2026-01-01T10:00:00Z",
            context="beverage preference",
            messages=[
                {"role": "user", "content": "I prefer tea in the afternoon."},
                {"role": "assistant", "content": "Recorded."},
            ],
        )
    ]

    result = client.pipeline(
        bank_id="vendor-smoke-test",
        sessions=sessions,
        question="What drink does the user prefer in the afternoon?",
        create_bank=True,
        reset_bank=True,
        bank_profile={
            "retain_mission": "Extract persistent user preferences and updates.",
            "reflect_mission": "Answer with the current user state grounded in recalled memory.",
        },
    )

    print(result.to_dict())


if __name__ == "__main__":
    main()
