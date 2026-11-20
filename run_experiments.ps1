param(
    [string]$OutputDir = 'outputs',
    [ValidateRange(0, 2147483647)]
    [int]$EpochsOverride = 0
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$scheduled = @(
    '--optimizer', 'adamw', '--scheduler', 'cosine',
    '--weight-decay', '0.0005', '--label-smoothing', '0.05'
)
$experiments = @(
    @{
        Name = 'exp0_baseline'
        Epochs = 30
        Options = @(
            '--arch', 'baseline', '--optimizer', 'adam', '--scheduler', 'none',
            '--weight-decay', '0.0001', '--label-smoothing', '0',
            '--dropout', '0.5', '--aug-level', 'standard'
        )
    }
    @{
        Name = 'exp1_scheduler'
        Epochs = 50
        Options = $scheduled + @('--arch', 'baseline', '--dropout', '0.5', '--aug-level', 'standard')
    }
    @{
        Name = 'exp2_deeper_cnn'
        Epochs = 50
        Options = $scheduled + @('--arch', 'deeper', '--dropout', '0.3', '--aug-level', 'standard')
    }
    @{
        Name = 'exp3_over_reg'
        Epochs = 50
        Options = $scheduled + @('--arch', 'deeper', '--dropout', '0.7', '--aug-level', 'heavy')
    }
)

Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($experiment in $experiments) {
        $epochs = if ($EpochsOverride -gt 0) { $EpochsOverride } else { $experiment.Epochs }
        $directory = Join-Path $OutputDir $experiment.Name
        $options = $experiment.Options
        Write-Host "Running $($experiment.Name) for $epochs epochs"
        & $python train.py --epochs $epochs --batch-size 128 --seed 42 --lr 0.001 --output-dir $directory @options
        if ($LASTEXITCODE -ne 0) { throw "Training failed in $($experiment.Name)" }
        & $python evaluate.py --output-dir $directory
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed in $($experiment.Name)" }
    }

    & $python compare.py --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) { throw "Comparison failed" }
}
finally {
    Pop-Location
}
