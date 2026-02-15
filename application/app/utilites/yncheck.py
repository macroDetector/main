def yncheck(user_input:str) -> bool:
    import time
    if user_input not in ['y', 'n']:
        print(f"❌ 잘못된 입력입니다 ('{user_input}').")
        for i in range(3, 0, -1):
            print(f"⚠️ {i}초 후 프로그램이 종료됩니다...", end="\r")
            time.sleep(1)
        print("\nBye! 👋")
        return False

    return True