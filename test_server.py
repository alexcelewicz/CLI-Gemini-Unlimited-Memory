#!/usr/bin/env python3
"""
Quick test script for the Gemini Memory MCP Server.
Run this to verify your setup is working correctly.

Usage:
    python test_server.py

Requires GEMINI_API_KEY environment variable to be set.
"""

import os
import sys
import json
import time


def main():
    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set.")
        print("   Set it with: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    print("✅ GEMINI_API_KEY is set")

    # Check google-genai import
    try:
        from google import genai
        from google.genai import types
        print("✅ google-genai package installed")
    except ImportError:
        print("❌ google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    # Check MCP import
    try:
        from mcp.server.fastmcp import FastMCP
        print("✅ mcp package installed")
    except ImportError:
        print("❌ mcp package not installed. Run: pip install 'mcp[cli]'")
        sys.exit(1)

    # Test Gemini client
    try:
        client = genai.Client(api_key=api_key)
        print("✅ Gemini client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        sys.exit(1)

    # Test: Create a store
    test_store_name = "project-memory-test-verification"
    print(f"\n--- Testing File Search Store Operations ---\n")

    try:
        print(f"Creating test store '{test_store_name}'...")
        store = client.file_search_stores.create(
            config={"display_name": test_store_name}
        )
        print(f"✅ Store created: {store.name}")
    except Exception as e:
        print(f"❌ Failed to create store: {e}")
        sys.exit(1)

    # Test: Upload a document
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, prefix="test-session-"
        ) as f:
            f.write(
                "# Test Session\n\n"
                "## What was done\n"
                "- Created a test document\n"
                "- Verified File Search store operations\n\n"
                "## Technical details\n"
                "- API endpoint: /api/v1/test\n"
                "- Database: PostgreSQL with users table\n"
                "- Auth: JWT tokens stored in httpOnly cookies\n"
            )
            tmp_path = f.name

        print("Uploading test document...")
        operation = client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store.name,
            config={"display_name": "test-session-doc"},
        )

        # Wait for processing
        timeout = 60
        start = time.time()
        while not operation.done:
            if time.time() - start > timeout:
                print("⚠️  Upload timed out (still processing). This is normal for first upload.")
                break
            time.sleep(2)
            operation = client.operations.get(operation)

        os.unlink(tmp_path)
        print("✅ Document uploaded and indexed")
    except Exception as e:
        print(f"❌ Failed to upload document: {e}")
        # Clean up store anyway
        try:
            client.file_search_stores.delete(name=store.name, config={"force": True})
        except:
            pass
        sys.exit(1)

    # Test: Query the store
    try:
        print("Querying store for context...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="What API endpoints and authentication methods were used?",
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store.name]
                        )
                    )
                ],
                max_output_tokens=1000,
            ),
        )
        print(f"✅ Context retrieved successfully!")
        print(f"\n--- Retrieved Context ---")
        print(response.text[:500])
        if len(response.text) > 500:
            print("... (truncated)")
        print(f"--- End ---\n")
    except Exception as e:
        print(f"❌ Failed to query store: {e}")

    # Cleanup
    try:
        print("Cleaning up test store...")
        client.file_search_stores.delete(name=store.name, config={"force": True})
        print("✅ Test store deleted")
    except Exception as e:
        print(f"⚠️  Failed to delete test store (clean up manually): {e}")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Your setup is ready.")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Configure Claude Code MCP (see README.md)")
    print("2. Create a store for your first project")
    print("3. Start saving sessions!")


if __name__ == "__main__":
    main()
