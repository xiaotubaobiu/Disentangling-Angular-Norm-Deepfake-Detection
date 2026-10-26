# Parallel fresh source-verification candidate

Operational experiment under the approved Detector v2 plan and performance authorization. Candidate lives outside the frozen repository and is not imported by production. No GPU, target inference, gate writer, training/configuration edit, or source-best dispatch.

Measured reference: original CLIP allfive source ledger, 10 independently accepted E30 models, 371.99 seconds. Prepare a maximum-two-thread collector that calls the ORIGINAL complete verifier afresh for every model on every invocation. Feed only these fresh receipts into the original allfive routine. Compare every returned ledger value with the serial reference except its top-level UTC. Bind config, registry, code, completion/seal inputs before/after. Any failure propagates without an output or partial qualification.

TDD: fresh file-content observations across calls; true bounded overlap; failures and invalid concurrency; one-use context-bound lookup; exact nested-ledger equality. Run the complete candidate suite and a source-only real CLIP serial/parallel equivalence benchmark. Independent review before any production integration. Candidate acceptance is not a H target gate or cache acceptance.
