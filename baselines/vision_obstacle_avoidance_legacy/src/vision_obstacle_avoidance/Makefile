# vision_obstacle_avoidance —— Makefile
# ============================================================================
# 用法:
#   make install         一键部署（pip + systemd + desktop）
#   make run             运行主程序
#   make dev             安装依赖并运行
#   make clean           清理临时文件
#   make uninstall       完整卸载
# ============================================================================

PYTHON     := python3
PIP        := pip
PROJECT    := $(shell pwd)

.PHONY: help install install-pip install-systemd install-desktop install-logrotate \
        run dev clean uninstall

help: ## 显示帮助信息
	@echo "vision_obstacle_avoidance —— Linux 部署工具"
	@echo ""
	@echo "  make install          一键部署（推荐）"
	@echo "  make run              运行主程序"
	@echo "  make dev              安装依赖 + 运行"
	@echo "  make clean            清理临时文件"
	@echo "  make uninstall        完整卸载"
	@echo ""
	@echo "  make install-pip       仅安装 Python 依赖"
	@echo "  make install-systemd   仅安装 systemd 用户服务"
	@echo "  make install-desktop   仅安装 .desktop 快捷方式"
	@echo "  make install-logrotate 仅安装 logrotate 配置（需 sudo）"

install: install-pip install-systemd install-desktop ## 一键部署全部组件
	@echo ""
	@echo "============================================"
	@echo "  部署完成！"
	@echo "============================================"
	@echo ""
	@echo "  运行主程序:            make run"
	@echo "  后台服务模式:          systemctl --user start vision-obstacle-avoidance"
	@echo "  启动语音守护:          systemctl --user start voice-daemon"
	@echo "  启用开机自启:          systemctl --user enable vision-obstacle-avoidance"
	@echo ""

install-pip: ## 安装 Python 依赖
	@echo "[pip] 安装依赖..."
	$(PIP) install -r requirements.txt

install-systemd: ## 安装 systemd 用户服务
	@echo "[systemd] 安装用户服务..."
	@mkdir -p $(HOME)/.config/systemd/user
	@sed 's|__PROJECT_DIR__|$(PROJECT)|g' systemd/vision-obstacle-avoidance.service > $(HOME)/.config/systemd/user/vision-obstacle-avoidance.service
	@sed 's|__PROJECT_DIR__|$(PROJECT)|g' systemd/voice-daemon.service > $(HOME)/.config/systemd/user/voice-daemon.service
	@sed 's|__PROJECT_DIR__|$(PROJECT)|g' systemd/voice-gui.service > $(HOME)/.config/systemd/user/voice-gui.service
	@systemctl --user daemon-reload
	@echo "[systemd] 已安装 3 个服务"

install-desktop: ## 安装 .desktop 快捷方式
	@echo "[desktop] 安装快捷方式..."
	@mkdir -p $(HOME)/.local/share/applications
	@sed 's|__PROJECT_DIR__|$(PROJECT)|g' desktop/vision-obstacle-avoidance.desktop > $(HOME)/.local/share/applications/vision-obstacle-avoidance.desktop
	@sed 's|__PROJECT_DIR__|$(PROJECT)|g' desktop/voice-gui.desktop > $(HOME)/.local/share/applications/voice-gui.desktop
	@sed 's|__PROJECT_DIR__|$(PROJECT)|g' desktop/voice-daemon.desktop > $(HOME)/.local/share/applications/voice-daemon.desktop
	@update-desktop-database $(HOME)/.local/share/applications/ 2>/dev/null || true
	@echo "[desktop] 已安装 3 个快捷方式"

install-logrotate: ## 安装 logrotate 配置（需 sudo）
	@echo "[logrotate] 安装日志轮转..."
	@sudo cp logrotate/vision-obstacle-avoidance /etc/logrotate.d/
	@sudo chmod 644 /etc/logrotate.d/vision-obstacle-avoidance
	@echo "[logrotate] 已安装"

run: ## 运行主程序
	$(PYTHON) main.py

dev: install-pip ## 安装依赖并运行
	$(PYTHON) main.py

analyze: ## 分析测试视频（用法: make analyze VIDEO=test.mp4）
	$(PYTHON) analyze_test.py $(VIDEO)

clean: ## 清理运行时文件
	@echo "[clean] 清理..."
	@-rm -rf logs/ videos/ .tmp/ 2>/dev/null || true
	@-rm -f .voice_*.wav /tmp/voice_*.pid 2>/dev/null || true
	@-find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "[clean] 完成"

uninstall: ## 完整卸载
	@echo "[uninstall] 停止服务..."
	@-systemctl --user stop vision-obstacle-avoidance voice-daemon voice-gui 2>/dev/null || true
	@-systemctl --user disable vision-obstacle-avoidance voice-daemon voice-gui 2>/dev/null || true
	@echo "[uninstall] 移除文件..."
	@-rm -f $(HOME)/.config/systemd/user/vision-obstacle-avoidance.service
	@-rm -f $(HOME)/.config/systemd/user/voice-daemon.service
	@-rm -f $(HOME)/.config/systemd/user/voice-gui.service
	@-rm -f $(HOME)/.local/share/applications/vision-obstacle-avoidance.desktop
	@-rm -f $(HOME)/.local/share/applications/voice-gui.desktop
	@-rm -f $(HOME)/.local/share/applications/voice-daemon.desktop
	@-systemctl --user daemon-reload 2>/dev/null || true
	@$(MAKE) clean
	@echo "[uninstall] 卸载完成"
