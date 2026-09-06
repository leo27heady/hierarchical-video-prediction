$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOutput = Join-Path $taskRoot 'output/pdf'
$taskDocx = Join-Path $taskOutput 'DISSERTATION_UKRAINIAN_HIERARCHY_2026-09-06.docx'
$taskPdf = Join-Path $taskOutput 'DISSERTATION_UKRAINIAN_HIERARCHY_2026-09-06.pdf'
$taskStaging = Join-Path $taskOutput ('review-export-' + [guid]::NewGuid().ToString() + '.pdf')
$taskWord = New-Object -ComObject Word.Application
$taskWord.Visible = $false
$taskWord.DisplayAlerts = 0
$taskWord.AutomationSecurity = 3
$taskDocument = $null
try {
    $taskDocument = $taskWord.Documents.Open($taskDocx, $false, $true)
    $null = $taskDocument.Fields.Update()
    foreach ($taskContents in $taskDocument.TablesOfContents) { $taskContents.Update() }
    $taskDocument.Repaginate()
    foreach ($taskContents in $taskDocument.TablesOfContents) { $taskContents.UpdatePageNumbers() }
    # Export all pages for printing with heading bookmarks and working links.
    $taskDocument.ExportAsFixedFormat($taskStaging, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1)
} finally {
    if ($null -ne $taskDocument) { $taskDocument.Close(0) }
    $taskWord.Quit()
}
$taskAllowed = [IO.Path]::GetFullPath($taskOutput).TrimEnd('\') + '\'
foreach ($taskPath in @($taskStaging, $taskPdf)) {
    if (-not [IO.Path]::GetFullPath($taskPath).StartsWith($taskAllowed, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Export path escaped its intended output directory'
    }
}
Move-Item -LiteralPath $taskStaging -Destination $taskPdf -Force
Write-Output $taskPdf
