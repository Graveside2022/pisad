#!/usr/bin/env python3
"""
Test script to verify MAVLink connectivity between PISAD and ASV Drones GBS
"""

import sys
import time
from pathlib import Path

# Add the src directory to Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pymavlink import mavutil
    print("✅ pymavlink imported successfully")
except ImportError as e:
    print(f"❌ Failed to import pymavlink: {e}")
    sys.exit(1)

def test_gbs_connection():
    """Test connection to ASV Drones GBS server."""
    
    print("Starting GBS connection test...")
    
    try:
        # Connect to GBS GUI port
        connection_string = "tcp:127.0.0.1:7341"
        print(f"Connecting to GBS at {connection_string}...")
        
        # Create MAVLink connection
        master = mavutil.mavlink_connection(
            connection_string,
            source_system=1,
            source_component=191  # MAV_COMP_ID_ONBOARD_COMPUTER
        )
        
        # Wait for first heartbeat
        print("Waiting for heartbeat from GBS...")
        msg = master.wait_heartbeat(timeout=10)
        if msg:
            print(f"✅ Heartbeat received from system {msg.get_srcSystem()}, component {msg.get_srcComponent()}")
            print(f"   Type: {msg.type}, Autopilot: {msg.autopilot}")
        else:
            print("❌ No heartbeat received within 10 seconds")
            return False
        
        # Send our own heartbeat
        print("Sending heartbeat...")
        master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,  # type
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,       # autopilot
            0,                                            # base_mode
            0,                                            # custom_mode
            mavutil.mavlink.MAV_STATE_ACTIVE             # system_status
        )
        
        # Listen for messages for 10 seconds
        print("Listening for MAVLink messages...")
        start_time = time.time()
        msg_count = 0
        
        while time.time() - start_time < 10:
            msg = master.recv_match(timeout=1.0)
            if msg:
                msg_count += 1
                if msg_count <= 5:  # Show first 5 messages
                    print(f"   Received: {msg.get_type()} from system {msg.get_srcSystem()}")
        
        print(f"✅ Received {msg_count} MAVLink messages in 10 seconds")
        
        if msg_count > 0:
            print("✅ GBS connection test completed successfully!")
            return True
        else:
            print("❌ No messages received from GBS")
            return False
            
    except Exception as e:
        print(f"❌ GBS connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_gbs_connection()