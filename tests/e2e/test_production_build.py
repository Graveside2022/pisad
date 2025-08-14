#!/usr/bin/env python3
"""
End-to-end test for production build functionality.
Tests that the built frontend can connect to backend properly.
"""

import asyncio
import json
import sys
from urllib.request import urlopen

import httpx
import websockets


async def test_production_build():
    """Test production build serves correctly and all features work."""
    results = {"passed": [], "failed": []}

    # Test 1: Frontend serves correctly
    try:
        with urlopen("http://localhost:4173/") as response:
            html = response.read().decode()
            if "<!doctype html>" in html.lower():
                results["passed"].append("Frontend HTML serves correctly")
            else:
                results["failed"].append("Frontend HTML structure incorrect")
    except Exception as e:
        results["failed"].append(f"Frontend serving failed: {e}")

    # Test 2: Backend API responds
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/api/system/status")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    results["passed"].append("Backend API responds correctly")
                else:
                    results["failed"].append("Backend API response invalid")
            else:
                results["failed"].append(f"Backend API returned {response.status_code}")
        except Exception as e:
            results["failed"].append(f"Backend API failed: {e}")

    # Test 3: WebSocket connection works
    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            # Wait for connection message
            message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            data = json.loads(message)
            if data.get("type") == "connection":
                results["passed"].append("WebSocket connection established")
            else:
                results["failed"].append("WebSocket connection message invalid")
    except Exception as e:
        results["failed"].append(f"WebSocket connection failed: {e}")

    # Test 4: CORS headers present
    async with httpx.AsyncClient() as client:
        try:
            response = await client.options(
                "http://localhost:8000/api/system/status",
                headers={"Origin": "http://localhost:4173"},
            )
            if "access-control-allow-origin" in response.headers:
                results["passed"].append("CORS headers configured")
            else:
                results["failed"].append("CORS headers missing")
        except Exception as e:
            results["failed"].append(f"CORS check failed: {e}")

    # Print results
    print("\n=== Production Build Test Results ===")
    print(f"\nPassed ({len(results['passed'])}):")
    for test in results["passed"]:
        print(f"  ✓ {test}")

    if results["failed"]:
        print(f"\nFailed ({len(results['failed'])}):")
        for test in results["failed"]:
            print(f"  ✗ {test}")
        return False
    else:
        print("\n✅ All production build tests passed!")
        return True


if __name__ == "__main__":
    success = asyncio.run(test_production_build())
    sys.exit(0 if success else 1)
