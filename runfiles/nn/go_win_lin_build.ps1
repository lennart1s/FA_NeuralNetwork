$env:GOOS="windows"
$env:GOARCH="amd64"
go build
$env:GOOS="linux"
$env:GOARCH="arm"
$env:GOARM="7"
echo Building...
go build
echo Built!
