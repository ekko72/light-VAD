[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot "..\..\.."))
$manifestPath = Join-Path $scriptRoot "a_final_freeze_manifest.json"
$shaLedgerPath = Join-Path $scriptRoot "a_final_sha256.txt"
$verificationPath = Join-Path $scriptRoot "a_final_verification.json"
$failurePath = Join-Path $scriptRoot "a_final_verification_failure.json"
$forbiddenNamePattern = "a2_new_final_ood_manifest\.json$|(?:^|[\\/])NEW_FINAL_OOD(?:[\\/]|$)"

function Assert-Equal {
    param(
        [Parameter(Mandatory = $true)] [AllowNull()] $Actual,
        [Parameter(Mandatory = $true)] [AllowNull()] $Expected,
        [Parameter(Mandatory = $true)] [string] $Label
    )

    if ($Actual -ne $Expected) {
        throw "$Label mismatch: expected '$Expected', got '$Actual'."
    }
}

function Resolve-RepoPath {
    param(
        [Parameter(Mandatory = $true)] [string] $RelativePath
    )

    if ($RelativePath -match $forbiddenNamePattern) {
        throw "Forbidden OOD path encountered in freeze metadata: $RelativePath"
    }

    $fullPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $RelativePath))
    $rootPrefix = $repoRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes repository root: $RelativePath"
    }
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        throw "Required frozen file is missing: $RelativePath"
    }

    return $fullPath
}

function Get-Sha256 {
    param(
        [Parameter(Mandatory = $true)] [string] $RelativePath
    )

    $fullPath = Resolve-RepoPath -RelativePath $RelativePath
    return (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash
}

function Read-ShaLedger {
    param(
        [Parameter(Mandatory = $true)] [string] $Path
    )

    $entries = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if ($trimmed.Length -eq 0 -or $trimmed.StartsWith("#")) {
            continue
        }

        if ($trimmed -notmatch "^([0-9A-Fa-f]{64})  (.+)$") {
            throw "Malformed SHA256 ledger line: $line"
        }

        $hash = $Matches[1].ToUpperInvariant()
        $relativePath = $Matches[2]
        if ($relativePath -match $forbiddenNamePattern) {
            throw "Forbidden OOD path encountered in SHA256 ledger: $relativePath"
        }
        if ($entries.ContainsKey($relativePath)) {
            throw "Duplicate SHA256 ledger path: $relativePath"
        }
        $entries[$relativePath] = $hash
    }

    return $entries
}

function Assert-LedgerHash {
    param(
        [Parameter(Mandatory = $true)] [hashtable] $Ledger,
        [Parameter(Mandatory = $true)] [string] $RelativePath,
        [Parameter(Mandatory = $true)] [string] $ExpectedHash
    )

    if (-not $Ledger.ContainsKey($RelativePath)) {
        throw "SHA256 ledger is missing: $RelativePath"
    }
    Assert-Equal -Actual $Ledger[$RelativePath] -Expected $ExpectedHash.ToUpperInvariant() -Label "SHA256 ledger entry for $RelativePath"
    Assert-Equal -Actual (Get-Sha256 -RelativePath $RelativePath) -Expected $ExpectedHash.ToUpperInvariant() -Label "Actual SHA256 for $RelativePath"
}

try {
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Freeze manifest is missing."
    }
    if (-not (Test-Path -LiteralPath $shaLedgerPath -PathType Leaf)) {
        throw "SHA256 ledger is missing."
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $ledger = Read-ShaLedger -Path $shaLedgerPath

    $manifestRelative = "results/difficulty_adaptive_context/a_final_claim_freeze/a_final_freeze_manifest.json"
    $verifierRelative = "results/difficulty_adaptive_context/a_final_claim_freeze/verify_a_final_freeze.ps1"

    if (-not $ledger.ContainsKey($manifestRelative)) {
        throw "SHA256 ledger does not bind the freeze manifest."
    }
    if (-not $ledger.ContainsKey($verifierRelative)) {
        throw "SHA256 ledger does not bind the read-only verifier."
    }

    $manifestHash = Get-Sha256 -RelativePath $manifestRelative
    $verifierHash = Get-Sha256 -RelativePath $verifierRelative
    Assert-Equal -Actual $ledger[$manifestRelative] -Expected $manifestHash -Label "Freeze manifest ledger hash"
    Assert-Equal -Actual $ledger[$verifierRelative] -Expected $verifierHash -Label "Verifier ledger hash"

    Assert-Equal -Actual $manifest.manifest_id -Expected "A-FINAL-MECHANISM-CLAIM-FREEZE-v1" -Label "Manifest id"
    Assert-Equal -Actual $manifest.freeze_status -Expected "FROZEN_MECHANISM_EXPLORATION_CLOSED" -Label "Freeze status"
    Assert-Equal -Actual $manifest.protocol.protocol_id -Expected "A-FINAL-OOD-v1" -Label "Protocol id"
    Assert-Equal -Actual $manifest.protocol.status -Expected "FROZEN_BEFORE_NEW_FINAL_OOD_ACCESS" -Label "Protocol status"
    Assert-Equal -Actual $manifest.protocol.path -Expected "results/difficulty_adaptive_context/a_final_claim_freeze/a_final_ood_protocol.md" -Label "Protocol path"
    Assert-Equal -Actual $manifest.sha256_ledger_path -Expected "results/difficulty_adaptive_context/a_final_claim_freeze/a_final_sha256.txt" -Label "SHA256 ledger path"

    Assert-LedgerHash -Ledger $ledger -RelativePath $manifest.protocol.path -ExpectedHash $manifest.protocol.sha256
    Assert-LedgerHash -Ledger $ledger -RelativePath $manifest.claim_ledger.path -ExpectedHash $manifest.claim_ledger.sha256

    $statuses = @{
        "A-v1" = "CONDITIONAL_GO"
        "A-v2" = "CONDITIONAL_GO"
        "CAR" = "STRONG_CROSS_ARCH_REPLICATION"
        "E1" = "REMOTE_INFORMATION_CONDITIONAL"
        "E2" = "INCONCLUSIVE_OR_INVALID"
        "E3" = "ROBUST_VALUE_STRUCTURE"
        "E4" = "INCONCLUSIVE_OR_INVALID"
    }
    foreach ($key in $statuses.Keys) {
        Assert-Equal -Actual $manifest.official_statuses.$key -Expected $statuses[$key] -Label "Official status $key"
    }

    Assert-Equal -Actual $manifest.stop_controls.STOP_MECHANISM_EXPLORATION -Expected $true -Label "STOP_MECHANISM_EXPLORATION"
    Assert-Equal -Actual $manifest.stop_controls.NEXT_EXPERIMENT_AUTHORIZED -Expected $false -Label "NEXT_EXPERIMENT_AUTHORIZED"
    Assert-Equal -Actual $manifest.stop_controls.TRAINING_PERFORMED -Expected $false -Label "TRAINING_PERFORMED"
    Assert-Equal -Actual $manifest.stop_controls.NEW_FINAL_OOD_TOUCHED -Expected $false -Label "NEW_FINAL_OOD_TOUCHED"
    Assert-Equal -Actual $manifest.runtime_binding.state -Expected "UNBOUND_AT_FREEZE_NOT_ACCESSED" -Label "Runtime OOD binding state"
    Assert-Equal -Actual $manifest.runtime_binding.manifest_path -Expected $null -Label "Runtime OOD manifest path"
    Assert-Equal -Actual $manifest.runtime_binding.manifest_sha256 -Expected $null -Label "Runtime OOD manifest hash"

    Assert-Equal -Actual $manifest.final_method.method -Expected "A-v2 M2" -Label "Final method"
    Assert-Equal -Actual $manifest.final_method.primary_budget -Expected 0.05 -Label "Primary budget"
    Assert-Equal -Actual $manifest.final_method.decision_threshold -Expected 0.5 -Label "Decision threshold"
    Assert-Equal -Actual $manifest.final_method.comparator.name -Expected "frozen uncertainty M0" -Label "Comparator"
    Assert-Equal -Actual $manifest.final_method.comparator.score -Expected "-abs(short_score - 0.5)" -Label "Comparator score"
    Assert-Equal -Actual $manifest.final_method.router_score -Expected "sigmoid(primary R head)" -Label "Router score"

    $expectedSeeds = @(17, 23, 41, 59, 71)
    Assert-Equal -Actual $manifest.final_method.training_seeds.Count -Expected $expectedSeeds.Count -Label "Training seed count"
    for ($i = 0; $i -lt $expectedSeeds.Count; $i++) {
        Assert-Equal -Actual $manifest.final_method.training_seeds[$i] -Expected $expectedSeeds[$i] -Label "Training seed $i"
        Assert-Equal -Actual $manifest.final_method.checkpoints[$i].seed -Expected $expectedSeeds[$i] -Label "Checkpoint seed $i"
        Assert-LedgerHash -Ledger $ledger -RelativePath $manifest.final_method.checkpoints[$i].path -ExpectedHash $manifest.final_method.checkpoints[$i].sha256
    }

    Assert-Equal -Actual $manifest.data_contract.input -Expected "A-v2 X3" -Label "Input contract"
    Assert-Equal -Actual $manifest.data_contract.feature_dimension -Expected 330 -Label "Feature dimension"
    Assert-Equal -Actual $manifest.data_contract.causal_window_frames -Expected 25 -Label "Causal window"
    Assert-Equal -Actual $manifest.data_contract.normalization.fit_scope -Expected "TRAIN only" -Label "Normalization fit scope"
    Assert-Equal -Actual $manifest.data_contract.normalization.ddof -Expected 0 -Label "Normalization ddof"
    Assert-Equal -Actual $manifest.data_contract.split_seed -Expected 20260920 -Label "Split seed"
    Assert-Equal -Actual $manifest.data_contract.split.TRAIN.speakers -Expected 24 -Label "TRAIN speaker count"
    Assert-Equal -Actual $manifest.data_contract.split.TRAIN.frames -Expected 313993 -Label "TRAIN frame count"
    Assert-Equal -Actual $manifest.data_contract.split.DEV.speakers -Expected 8 -Label "DEV speaker count"
    Assert-Equal -Actual $manifest.data_contract.split.DEV.frames -Expected 126522 -Label "DEV frame count"
    Assert-Equal -Actual $manifest.data_contract.split.INTERNAL_TEST.speakers -Expected 8 -Label "INTERNAL_TEST speaker count"
    Assert-Equal -Actual $manifest.data_contract.split.INTERNAL_TEST.frames -Expected 113017 -Label "INTERNAL_TEST frame count"

    Assert-Equal -Actual $manifest.evaluation_contract.primary_endpoint -Expected "Delta_U_primary = mean_s(U(M2, s, 0.05) - U(M0, s, 0.05))" -Label "Primary endpoint"
    Assert-Equal -Actual $manifest.evaluation_contract.aggregate_test.resamples -Expected 2000 -Label "Bootstrap resamples"
    Assert-Equal -Actual $manifest.evaluation_contract.aggregate_test.seed -Expected 20260921 -Label "Bootstrap seed"
    Assert-Equal -Actual $manifest.evaluation_contract.aggregate_test.interval -Expected "two-sided 95 percent" -Label "Bootstrap interval"
    Assert-Equal -Actual $manifest.evaluation_contract.allowed_subgroups.minimum_eligible_frames_for_g3 -Expected 1000 -Label "Minimum G3 frame count"

    Assert-Equal -Actual $manifest.frozen_gates.G1.frozen_a_v2_result -Expected "FAIL" -Label "Frozen G1 result"
    Assert-Equal -Actual $manifest.frozen_gates.G2.frozen_a_v2_result -Expected "PASS" -Label "Frozen G2 result"
    Assert-Equal -Actual $manifest.frozen_gates.G3.frozen_a_v2_result -Expected "PASS" -Label "Frozen G3 result"
    Assert-Equal -Actual $manifest.frozen_gates.G4.frozen_a_v2_result -Expected "PASS" -Label "Frozen G4 result"
    Assert-Equal -Actual $manifest.frozen_gates.G5.frozen_a_v2_result -Expected "PASS" -Label "Frozen G5 result"

    foreach ($artifact in $manifest.frozen_artifacts) {
        Assert-LedgerHash -Ledger $ledger -RelativePath $artifact.path -ExpectedHash $artifact.sha256
    }
    foreach ($code in $manifest.evaluation_code) {
        Assert-LedgerHash -Ledger $ledger -RelativePath $code.path -ExpectedHash $code.sha256
    }

    foreach ($relativePath in $ledger.Keys) {
        if ($relativePath -ne $verifierRelative) {
            Assert-Equal -Actual (Get-Sha256 -RelativePath $relativePath) -Expected $ledger[$relativePath] -Label "Actual SHA256 for ledger path $relativePath"
        }
    }

    $result = [ordered]@{
        verification_id = "A-FINAL-FREEZE-VERIFY-v1"
        verified_utc = [DateTime]::UtcNow.ToString("o")
        status = "PASS"
        manifest_path = $manifestRelative
        manifest_sha256 = $manifestHash
        protocol_hash = $manifest.protocol.sha256
        verifier_path = $verifierRelative
        verifier_sha256 = $verifierHash
        frozen_artifact_count = $manifest.frozen_artifacts.Count
        evaluation_code_count = $manifest.evaluation_code.Count
        checkpoint_count = $manifest.final_method.checkpoints.Count
        ledger_entry_count = $ledger.Count
        official_statuses = $manifest.official_statuses
        frozen_gate_results = [ordered]@{
            G1 = $manifest.frozen_gates.G1.frozen_a_v2_result
            G2 = $manifest.frozen_gates.G2.frozen_a_v2_result
            G3 = $manifest.frozen_gates.G3.frozen_a_v2_result
            G4 = $manifest.frozen_gates.G4.frozen_a_v2_result
            G5 = $manifest.frozen_gates.G5.frozen_a_v2_result
        }
        STOP_MECHANISM_EXPLORATION = $manifest.stop_controls.STOP_MECHANISM_EXPLORATION
        NEW_FINAL_OOD_TOUCHED = $manifest.stop_controls.NEW_FINAL_OOD_TOUCHED
        TRAINING_PERFORMED = $manifest.stop_controls.TRAINING_PERFORMED
        NEXT_EXPERIMENT_AUTHORIZED = $manifest.stop_controls.NEXT_EXPERIMENT_AUTHORIZED
    }

    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $verificationPath -Encoding utf8
    $result | ConvertTo-Json -Depth 8
}
catch {
    $failure = [ordered]@{
        verification_id = "A-FINAL-FREEZE-VERIFY-v1"
        verified_utc = [DateTime]::UtcNow.ToString("o")
        status = "FAIL"
        error = $_.Exception.Message
        NEW_FINAL_OOD_TOUCHED = $false
        TRAINING_PERFORMED = $false
    }
    $failure | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $failurePath -Encoding utf8
    Write-Error $_.Exception.Message
    exit 1
}
