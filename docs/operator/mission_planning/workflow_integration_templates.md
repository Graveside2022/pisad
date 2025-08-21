# Mission Planning Workflow Integration Templates

## Overview

This document provides standardized workflow templates for integrating PISAD RF operations into Mission Planner-based SAR missions. These templates ensure consistent, thorough mission planning while maintaining compliance with PRD safety requirements and operational efficiency standards.

**Integration Philosophy:** Seamless incorporation of RF homing capabilities into standard SAR workflows without disrupting established Mission Planner procedures.

---

## **TEMPLATE 1: STANDARD SAR MISSION WITH RF HOMING**

### Mission Overview Template

**Mission Identification:**
- Mission ID: [SAR-YYYY-MMDD-###]
- Mission Type: Standard Emergency Beacon SAR with RF Homing
- Priority Level: [Routine/Urgent/Emergency/Life Threatening]
- Estimated Duration: [Flight hours including search and homing phases]
- Aircraft Assignment: [Tail number and PISAD system ID]

**Target Information:**
- Beacon Type: [406MHz ELT/PLB/EPIRB or Custom frequency]
- Expected Signal Characteristics: [Power level, modulation, transmission pattern]
- Last Known Position: [GPS coordinates with confidence radius]
- Search Area: [Primary search area boundaries and alternate areas]
- Environmental Factors: [Terrain, weather, RF interference assessment]

### Mission Planning Workflow

**Phase 1: Pre-Planning (T-24 to T-2 hours)**
□ Weather Assessment and Go/No-Go Decision
  - Wind: ≤25kt sustained, ≤35kt gusts (PISAD operational limits)
  - Visibility: ≥3sm for visual references during homing
  - Precipitation: No severe icing conditions affecting antenna
  - Temperature: Within -20°C to +60°C operational range

□ Aircraft and Equipment Preparation
  - Aircraft airworthiness and fuel planning for extended search
  - PISAD system functionality verification (15-minute startup test)
  - Mission Planner configuration with current PISAD parameters
  - Communication equipment and backup navigation systems

□ Regulatory and Coordination
  - ATC coordination for search area and altitude assignments
  - Frequency coordination with other emergency services
  - SAR incident command notification and authorization
  - Geofence configuration for operational area boundaries

**Phase 2: Mission Planning (T-2 to T-30 minutes)**
□ Mission Planner Setup
  - Load search area boundaries and waypoint structure
  - Configure PISAD RF parameters based on beacon type
  - Set up telemetry displays for RF status monitoring
  - Verify emergency abort procedures and alternative landing sites

□ Search Pattern Generation
  - Primary search pattern: [Expanding square/Track crawl/Sector search]
  - Pattern altitude: [AGL considering terrain and RF propagation]
  - Search spacing: [Based on beacon power and detection range]
  - Homing transition criteria: [Signal strength and confidence thresholds]

□ Safety Configuration
  - Emergency return-to-launch parameters and fuel reserves
  - Lost communication procedures and rally points
  - Weather alternate plans and abort criteria
  - Battery management plan for extended operations

**Phase 3: Pre-Flight (T-30 to T-0 minutes)**
□ Final System Verification
  - PISAD parameter synchronization confirmation
  - RF health status verification (>70% minimum)
  - Telemetry link quality check (<2% packet loss)
  - Emergency disable command functionality test

□ Crew Briefing
  - Mission objectives and search area boundaries
  - RF homing activation criteria and procedures
  - Emergency abort procedures and authority hierarchy
  - Communication protocols and decision points

□ Authority and Authorization
  - SAR incident command final mission approval
  - Pilot-in-command briefing and flight plan acceptance
  - Final weather check and go/no-go decision
  - Mission Planner flight plan upload and verification

### Mission Execution Workflow

**Transit to Search Area:**
□ Navigate to search area using standard Mission Planner autopilot
□ Establish communication with SAR incident command
□ Begin PISAD RF monitoring (passive detection mode)
□ Monitor system health parameters and telemetry quality
□ Verify geofence boundaries and emergency procedures

**Search Phase:**
□ Execute planned search pattern maintaining PISAD RF monitoring
□ Monitor for signal detection indicators:
  - PISAD_SIG_CONF >30% for initial detection
  - PISAD_RSSI increasing above noise floor
  - PISAD_BEARING showing consistent direction
□ Coordinate with ground teams and other aircraft
□ Document search progress and any signals detected

**Homing Activation Decision Point:**
Decision Criteria for Homing Activation:
□ Signal confidence (PISAD_SIG_CONF) ≥50%
□ Bearing confidence (PISAD_BEAR_CONF) ≥70%
□ Aircraft in GUIDED mode with stable flight
□ Battery level >25% remaining
□ Weather conditions within operational limits
□ No higher priority emergency situations

**Homing Phase:**
□ Execute MAV_CMD_USER_1 (Param1=1) to activate homing
□ Verify PISAD_HOMING_STATE transitions to 1 (Armed) then 2 (Active)
□ Monitor homing progress through telemetry parameters
□ Maintain continuous awareness of aircraft position and safety
□ Be prepared for immediate manual override if required

**Target Acquisition:**
□ Signal strength plateau indicates target proximity
□ PISAD_HOMING_STATE automatically returns to 0 (Disabled)
□ Conduct visual search of target area
□ Document target location with GPS coordinates
□ Photograph target area for SAR documentation
□ Coordinate with ground rescue teams for target recovery

---

## **TEMPLATE 2: EMERGENCY/TIME-CRITICAL SAR MISSION**

### Rapid Deployment Template

**Expedited Planning (T-30 to T-0 minutes):**
□ Mission authorization and immediate deployment approval
□ Aircraft selection based on readiness status and PISAD capability
□ Abbreviated weather briefing focusing on operational safety
□ Rapid PISAD parameter configuration using emergency profiles

Emergency Profile Selection:
□ Profile 0: 406MHz Emergency Beacon (standard ELT/PLB)
□ Profile 1: 121.5MHz Aviation Emergency (aircraft ELT backup)
□ Profile 2: 162MHz Marine Emergency (EPIRB signals)
□ Custom: [Specific frequency if known from incident information]

**Streamlined Pre-Flight:**
□ PISAD system health verification (5-minute minimum check)
□ Mission Planner quick setup with emergency search parameters
□ Abbreviated crew briefing focusing on safety and mission critical items
□ Direct transit clearance and abbreviated flight plan

### Time-Critical Execution Modifications

**Accelerated Search Pattern:**
□ Direct transit to most probable target area
□ Reduced search pattern spacing for faster area coverage
□ Lower altitude if safe and effective for signal detection
□ Continuous RF monitoring during transit phase
□ Immediate homing activation upon signal detection

**Emergency Decision Criteria:**
Modified Homing Activation (Emergency Missions):
□ Signal confidence (PISAD_SIG_CONF) ≥40% (reduced from standard 50%)
□ Bearing confidence (PISAD_BEAR_CONF) ≥60% (reduced from standard 70%)
□ Aircraft stability and safety still paramount
□ Battery management with expedited decision timeline
□ Weather safety limits maintained despite time pressure

---

## **TEMPLATE 3: TRAINING MISSION INTEGRATION**

### Training Mission Planning Template

**Training Objectives Integration:**
□ Primary Objective: [Specific PISAD competency being trained]
□ Secondary Objectives: [Supporting skills and Mission Planner integration]
□ Safety Objectives: [Emergency procedures and safety protocol practice]
□ Assessment Criteria: [Specific performance standards being evaluated]
□ Instructor Authority: [Training safety authority and abort criteria]

**Training-Specific Configuration:**
□ Training beacon setup with known location and characteristics
□ Practice area boundaries with adequate safety margins
□ Student competency level appropriate challenge configuration
□ Observer aircraft coordination if required for advanced training
□ Data recording setup for performance analysis and debrief

### Training Mission Execution

**Pre-Training Brief:**
□ Training objectives and expected performance standards
□ Safety procedures and instructor authority hierarchy
□ Emergency abort procedures and training termination criteria
□ Mission Planner setup and PISAD parameter configuration
□ Scenario setup and expected progression

**Training Execution Monitoring:**
□ Continuous performance assessment against training standards
□ Safety monitoring with immediate intervention capability
□ Real-time coaching and guidance as appropriate for competency level
□ Documentation of training progression and performance
□ Immediate feedback and correction for critical errors

**Post-Training Analysis:**
□ Immediate debrief discussion of performance and decision points
□ Detailed performance analysis using recorded telemetry data
□ Competency assessment documentation for certification records
□ Training progression recommendations and development planning
□ Lessons learned capture for training program improvement

---

## **WORKFLOW CUSTOMIZATION GUIDELINES**

### Mission-Specific Adaptations

**Environmental Adaptations:**
- Mountain SAR: Increased altitude considerations, terrain masking effects
- Water SAR: Marine frequency priorities, coordination with Coast Guard
- Urban SAR: Interference management, ATC coordination complexity
- Wilderness SAR: Extended range planning, fuel management emphasis

**Equipment Adaptations:**
- HackRF Performance Variations: Power and sensitivity considerations
- Antenna Configuration: Directional vs omnidirectional based on mission
- Multiple PISAD Systems: Coordination between different PISAD-equipped aircraft
- Integration Limitations: Workarounds for partial system availability

### Regulatory Adaptations

**Frequency Coordination:**
- Emergency Services Coordination: Police, fire, EMS frequency protection
- Military Airspace: Additional coordination requirements and restrictions
- International Operations: Cross-border frequency and operational coordination
- Special Use Airspace: Restricted areas and coordination protocols

**Documentation Requirements:**
- Incident Command System: ICS integration and reporting requirements
- Aviation Authorities: Flight plan and operational reporting requirements
- SAR Organizations: Organizational documentation and performance reporting
- Insurance and Liability: Mission documentation for coverage and analysis

---

**Document Version:** 1.0  
**Last Updated:** 2025-08-21  
**Distribution:** All PISAD operators, mission planners, SAR coordinators