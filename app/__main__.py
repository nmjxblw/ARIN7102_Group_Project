if __name__ == "__main__":
    print(f"{__package__}.{__name__} 被作为主程序运行，启动 launcher 模块...")

    profile_filename = f"app_runtime_result.stats"

    # 系统/第三方模块导入
    # import cProfile

    # 本地模块导入
    # from launcher_module import run

    # cProfile.run(
    #     statement="run()",
    #     filename=profile_filename,
    #     sort="cumulative",
    # )

    # import pstats

    # text_filename = "profile_readable.txt"
    # with open(text_filename, "w", encoding="utf-8") as f:
    #     ps = pstats.Stats(profile_filename, stream=f)
    #     ps.strip_dirs().sort_stats("cumulative").print_stats()
    import dataset_module

    # dataset_module.download_and_open_datasets()
    dataset_module.generate_visualize_data_frame()

    # import matplotlib.font_manager

    # fonts = matplotlib.font_manager.findSystemFonts()

    # for font in sorted(fonts[:]):  # 只显示前10个
    #     print(font)
