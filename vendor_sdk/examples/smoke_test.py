from hms_vendor_sdk import HMSVendorClient, SessionRecord


def main() -> None:
    client = HMSVendorClient.from_env()

    sessions = [
        SessionRecord(
            session_id="demo-session-1",
            timestamp="2024-03-01T10:00:00Z",
            context="shopping reminders",
            messages=[
                {"role": "user", "content": "I need to pick up my blazer from dry cleaning."},
                {"role": "assistant", "content": "Recorded."},
                {"role": "user", "content": "I also need to return the boots that are too small."},
            ],
        )
    ]

    result = client.pipeline(
        bank_id="vendor-smoke-test",
        sessions=sessions,
        question="How many items do I need to pick up or return from a store?",
        question_date="2024-03-10T00:00:00Z",
        create_bank=True,
        reset_bank=True,
        bank_profile={
            "retain_mission": "Extract persistent user state, tasks, shopping actions, and updates.",
            "reflect_mission": "Answer with the current user state grounded in recalled memory.",
        },
    )

    print(result.to_dict())


if __name__ == "__main__":
    main()
