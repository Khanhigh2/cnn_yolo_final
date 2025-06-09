@echo off
echo ========================================
echo    HE THONG PHAT HIEN LOI
echo    Defect Detection System
echo ========================================
echo.
echo Dang khoi dong server...
echo Starting server...
echo.

cd /d "c:\project_root\web3"
D:\python.exe app.py

echo.
echo Server da dung. Nhan phim bat ky de thoat...
echo Server stopped. Press any key to exit...
pause > nul
