REPO_INFO = {
    "seatmap_distrib": {
        "purpose": (
            "Acts as the distribution gateway for seatmap and pricing flows between internal inventory systems (RES/DCS) "
            "and external channels (EDIFACT and cryptic). Combines SMP (Seatmap Pricing), SME (Seatmap EDIFACT), and "
            "SMC (Seatmap Cryptic) components to handle seatmap availability, pricing, and synchronization."
        ),
        "main_components": [
            "SME (Seatmap EDIFACT gateway handling seat availability and seatmap conversion)",
            "SMC (Seatmap Cryptic gateway for GDS and cryptic terminal requests)",
            "SMP (Seatmap Pricing layer integrating seat selection and pricing data)"
        ],
        "supported_messages": {
            # Seatmap flows
            "SMPREQ": "Seatmap pricing request",
            "SMPRES": "Seatmap pricing response",
            "SMPTTQ": "Seatmap pricing ticketing query",
            # Functional requests
            "HSFREQ": "Flight status fetch request",
            "TRMREQ": "Terminal request",
            # Seatmap enrichment and migration
            "PACVRR": "Passenger allocation conversion response",
            "PAPVRR": "Passenger allocation pricing validation response",
            "FTVREQ": "Flight transfer request",
            "PLADRR": "Place address request",
            # Functional and ancillary
            "FFLIPR": "Flight flip response",
            "AGAMRR": "Agent management response",
            "GENRES": "General response"
        },
        "core_responsibilities": [
            "Enrich and convert seatmap requests to the correct internal inventory system (RES/DCS).",
            "Handle both EDIFACT and cryptic (GDS) message parsing, encoding, and validation.",
            "Combine seat availability and pricing information via SMP integration.",
            "Distribute seatmap, template, and flight-related data across systems during migration or rebuilds.",
            "Coordinate flows with STM (seat management) and STE (seating distribution) for downstream processing."
        ],
    },
    "ste": {
        "purpose": (
            "Handles the distribution, enrichment, and response generation of seat-related EDIFACT messages "
            "between external systems (e.g., GDS, DCS) and internal backends (STM, DBA). "
            "Acts as the primary functional server for seating message routing and translation."
        ),
        "main_components": [
            "STE_FS-T (Functional Server for ISELGQ message routing and enrichment)",
            "STEBackend (core runtime managing backend context, flow routing, and PegProxy calls)",
            "STEPackage (seat response serialization and packaging service)",
        ],
        "supported_messages": {
            "ISELGQ": "Seat request message",
            "ISELGR": "Seat response message",
            "SPAXAR": "Passenger seat allocation request",
            "SPAXDR": "Passenger seat allocation display request",
            "SPAXAQ": "Passenger seat allocation query",
            "SPAXDQ": "Passenger seat allocation display query",
            "SACCRQ": "Ancillary charge request",
            "CSHSTQ": "Seat history request",
            "MROUGR": "Group seat movement request"
        },
        "core_responsibilities": [
            "Receive and interpret ISELGQ seat requests from distribution layer and route them to STM/DBA.",
            "Translate seat allocation, display, and update messages to internal format and back to EDIFACT.",
            "Coordinate with STM for seat allocation and DBA for seat history synchronization.",
            "Build and serialize response seatmaps with enriched context and availability data.",
            "Handle PegProxy communication for distributed and cross-application flows."
        ],
    },
    "stm_main": {
        "purpose": (
            "Acts as the central Seat Management (STM) system responsible for handling all seatmap-related "
            "inventory operations including seat allocation, reseating, rebuild, migration, and discrepancy correction. "
            "It integrates both synchronous (EDIFACT) and asynchronous (queued) seatmap flows and manages seatmap "
            "data consistency across RES and DCS systems."
        ),
        "main_components": [
            "SMD (Seatmap Display) – handles seatmap requests and responses from RES and DCS channels (SSMFRQ/R, SMPREQ/R).",
            "STG (Seating) – processes seat allocation, update, and ancillary seating messages from both RES and DCS (SPAX*, SCT*, SIGREQ).",
            "RST (Reseating) – manages reseating requests in RES and DCS environments, handling SRBRCQ, SRBRSQ, SCOIUQ, and related flows.",
            "RBC (Rebuild & Compare) – detects and resolves seatmap discrepancies, manages rebuild requests (SPNLRQ, SREBUQ, SCMPPQ, IRBRTQ).",
            "SMM (Seatmap Management) – handles template, configuration, ACV, and master seatmap management (SACV*, SCON*, SMIG*, SSMT*).",
            "FDS (Flight Date Seatmap) – manages flight date updates, redress processes, and demigration flows (CRUFRQ, CRUSRQ, CCAPUQ, FNOTSQ).",
            "FMT (Flight Migration Table) – handles migration/demigration of flight seatmap data (MFMARQ, MFMSRQ, MFMTSQ).",
            "TEC (Technical module) – manages partition cleanup, seatmap blob handling, and batch data maintenance (STM_LD_MAIN_BLOB, STM_REACC_INFO)."
        ],
        "supported_messages": {
            # Seatmap display
            "SSMFRQ": "Seatmap flight request",
            "SSMFRR": "Seatmap flight response",
            "SMPREQ": "Seatmap pricing request",
            "SMPRES": "Seatmap pricing response",
            "SGSMDQ": "Seatmap discrepancy query",
            "SSMPPQ": "Seatmap pricing query",
            # Seating and ancillary
            "SPAX*": "Passenger seat allocation messages",
            "SCT*": "Seat configuration messages",
            "SNCP*": "Seat notification messages",
            "SGST*": "Seat guest messages",
            "SCBSUQ": "Seat booking status update query",
            "SEOT*": "Seat event messages",
            "SIGREQ": "Seat inventory request",
            # Reseating
            "SRBRCQ": "Reseating build request",
            "SRBRSQ": "Reseating response",
            "SCOIUQ": "Seat change order update query",
            "SEREUQ": "Seat reallocation update query",
            "SPAXRQ": "Passenger seat allocation request",
            # Rebuild / Discrepancy
            "SPNLRQ": "Seatplan reload request",
            "SREBUQ": "Seatmap rebuild request",
            "IRBRTQ": "Internal rebuild trigger",
            "SCMPPQ": "Seatmap compare query",
            "SLKOEQ": "Seat lock operation request",
            "SFIDEQ": "Seat field edit request",
            # Seatmap management
            "SACV*": "Seat availability messages",
            "SCON*": "Seat configuration messages",
            "SMIG*": "Seat migration messages",
            "SSMT*": "Seat master template messages",
            "SSEDRQ": "Seat edit request",
            "SSMURQ": "Seat map update request",
            # Flight data management
            "CRUFRQ": "Flight redress request",
            "CRUSRQ": "Flight redress status request",
            "CCAPUQ": "Flight capacity update query",
            "FNOTSQ": "Flight notification request",
            "SDMGNQ": "Seat damage notification query",
            "MDAGPQ": "Migration data aggregation query"
        },
        "core_responsibilities": [
            "Manage seat allocation, reseating, and rebuild flows across RES and DCS systems.",
            "Process seatmap display, configuration, and migration requests through EDIFACT and queued interfaces.",
            "Maintain flight date-level seatmaps and synchronize across RES/DCS for operational consistency.",
            "Handle rebuild/discrepancy detection and automated resolution through RBC load item.",
            "Manage templates, ACV, configurations, and migration logic in SMM.",
            "Perform redress, demigration, and recovery tasks through FDS and FMT modules.",
            "Provide asynchronous job handling and housekeeping tasks through TEC and batch processes."
        ],
    },
    "fmt": {
        "purpose": (
            "Manages the Flight Migration Table (FMT) responsible for migration, demigration, "
            "and maintenance of seatmap and flight configuration data between legacy and new systems."
        ),
        "main_components": [
            "FMT_FS-T (functional server handling flight demigration services)",
            "FMT_CB-Purge (batch for purging old flight migration data)",
            "FMT_FQ-Replica (asynchronous process for migration replication and maintenance)"
        ],
        "supported_messages": {
            "MFMARQ": "Migration flight mapping request",
            "MFMSRQ": "Migration flight status request",
            "MFMTSQ": "Migration flight transfer status query",
            "MFMFRQ": "Migration flight forward request",
            "MSMTRQ": "Migration seat map transfer request"
        },
        "core_responsibilities": [
            "Process migration and demigration requests for flight seatmap data.",
            "Handle FMT maintenance operations like data replication and purge.",
            "Manage synchronization between RES and DCS for migrated flights.",
            "Ensure consistency and validation of demigrated flight data.",
            "Coordinate rebuild and recovery processes linked with STM."
        ],
    },
    "dfd": {
        "purpose": (
            "Manages the Discrepancy Framework for the DCS environment. It records, reports, "
            "and synchronizes seatmap discrepancies detected on the DCS side, ensuring "
            "alignment with operational processes and supporting analytics and recovery."
        ),
        "main_components": [
            "Record Builder – constructs structured discrepancy records from DCS seatmap data.",
            "Reporting – generates Excel/email reports summarizing discrepancies and rebuild statuses.",
            "Elastic Integration – forwards Kafka events and data snapshots to ElasticSearch for dashboards.",
            "Kafka Forwarder Daemon – runs background tasks for publishing rebuild metrics and status updates."
        ],
        "core_responsibilities": [
            "Detect and record DCS-side seatmap discrepancies reported by STM or external triggers.",
            "Run scheduled jobs (Record, Report) to consolidate and export discrepancy data.",
            "Forward rebuild and reporting metrics to ElasticSearch for operational monitoring.",
            "Coordinate with Kafka daemons for asynchronous event delivery and recovery metrics.",
            "Serve as DCS counterpart to DFR’s RES discrepancy framework, ensuring consistency of detection logic."
        ],
    },
    "dfr": {
        "purpose": (
            "Handles discrepancy detection, aggregation, filtering, rebuilding, and reporting of seatmap data "
            "within the RES environment. Ensures internal seatmap consistency after flight rebuilds or refreshes "
            "and supports airlines in identifying, classifying, and resolving seatmap issues through automated "
            "or manual workflows."
        ),
        "main_components": [
            "Aggregation – consolidates discrepancy data across flights and airlines for reporting and visualization.",
            "Filter – isolates actionable discrepancies based on airline parameters and rebuild priorities.",
            "Rebuild – triggers automated rebuilds of affected flights and monitors completion status.",
            "Seat Refresh – initiates targeted seat refreshes when rebuild is not required.",
            "Reporting – generates Excel/email reports and pushes summary data to Elastic for dashboards.",
            "Check Rebuild – validates rebuild outcomes and updates status accordingly.",
            "Kafka Forwarder – streams rebuild and reporting metrics to ElasticSearch for observability."
        ],
        "core_responsibilities": [
            "Detect and aggregate seatmap discrepancies using ElasticSearch as a centralized source of truth.",
            "Filter and prioritize discrepancies for automated rebuild or manual follow-up.",
            "Trigger rebuild and seat refresh flows through EDIFACT messages routed via PEG.",
            "Monitor job completion and validate rebuilt seatmaps via Check Rebuild jobs.",
            "Publish rebuild, reporting, and filtering metrics to Elastic for analytics.",
            "Provide reporting pipelines to generate Excel/email outputs for operational monitoring."
        ],
    },
    "dba": {
        "purpose": (
            "Acts as the central database schema and data management repository for seating applications. "
            "It defines and maintains all database tables, indexes, constraints, and cache loaders used by "
            "the STM, FMT, STE, DFR, and DFD systems. The DBA repository provides a single source of truth "
            "for persistent seatmap data, flight configurations, and history management."
        ),
        "main_components": [
            "Schema Definition – SQL scripts defining creation of all functional schemas (STM, FMT, COM, MIB, REF, LQS, etc.).",
            "Index and Constraint Management – manages performance-optimized access patterns for seatmap, ACV, and configuration data.",
            "Load Scripts – contains data population scripts for initializing reference and cache tables (EZT, EST, etc.).",
            "Component Structure – submodules for STM, FMT, COM, MIB, REF, LQS representing functional database areas."
        ],
        "core_responsibilities": [
            "Provide physical database structures used by STM, FMT, STE, and discrepancy frameworks (DFR, DFD).",
            "Define and version-control SQL schemas, indexes, triggers, and stored procedures for all seat-related components.",
            "Maintain cache and lookup tables used across RES/DCS systems for fast seat availability and rebuild validation.",
            "Support database operations required for seatmap rebuild, redress, history storage, and recovery flows.",
            "Serve as the foundation for the DBA layer in STM and other backends’ DBAdaptors."
        ],
    }
}