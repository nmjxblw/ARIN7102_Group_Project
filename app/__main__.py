if __name__ == "__main__":
    print(f"{__package__}.{__name__} 被作为主程序运行，启动 launcher 模块...")

    profile_filename = f"app_runtime_result.stats"

    # 本地模块导入
    from launcher_module import app_run

    app_run()
