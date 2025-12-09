@echo off
echo Updating CNCMate Repository...
git add .
set /p commit_msg="Enter commit message: "
git commit -m "%commit_msg%"
git push origin main
echo Done!
pause
