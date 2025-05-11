import configparser
import os

from selenium import webdriver
from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.remote_connection import FirefoxRemoteConnection
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxWebDriver
from selenium.webdriver.remote.client_config import ClientConfig
from webdriver_manager.firefox import GeckoDriverManager


# finds the path to the firefox default profile so that we can get it and use it with out webdriver
def get_default_firefox_profile() -> str:
    try:
        print("getting user settings profile")
        app_data = os.environ["APPDATA"]
        profiles_dir = os.path.join(app_data, "Mozilla", "Firefox", "Profiles")
        profiles_ini = os.path.join(app_data, "Mozilla", "Firefox", "profiles.ini")

        if not os.path.exists(profiles_ini):
            raise Exception("Firefox profiles.ini not found")

        config = configparser.ConfigParser()
        config.read(profiles_ini)

        default_profile = None
        for section in config.sections():
            if not section.startswith("Profile0"):
                continue

            default_profile = config[section]
            break

        if not default_profile:
            for section in config.sections():
                if not section.startswith("Profile"):
                    continue

                if config.get(section, "Default", fallback="0") == "1":
                    default_profile = config[section]
                    break

                profile_name = config.get(section, "Name", fallback="").lower()
                if "default" in profile_name:
                    default_profile = config[section]
                    break

        if not default_profile:
            raise Exception("No default Firefox profile found")

        is_relative = config.getboolean(section, "IsRelative", fallback=True)
        raw_path = default_profile["Path"]

        if is_relative:
            clean_path = raw_path.replace("\\", "/").split("/")[-1]
            full_path = os.path.join(profiles_dir, clean_path)
        else:
            full_path = os.path.normpath(raw_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Profile path invalid: {full_path}")

        return full_path

    except Exception as e:
        print(f"Error finding firefox default profile: {e}")
        exit()


# custom firefox webdrive in order to have a timeout longer than the default 120s
class CustomFirefoxWebDriver(FirefoxWebDriver):
    def __init__(
        self, timeout_seconds: int, options=None, service=None, keep_alive=True
    ):
        self.service = service if service else FirefoxWebDriver.Service()
        options = options if options else FirefoxWebDriver.Options()

        finder = DriverFinder(self.service, options)
        if finder.get_browser_path():
            options.binary_location = finder.get_browser_path()
            options.browser_version = None

        self.service.path = self.service.env_path() or finder.get_driver_path()
        self.service.start()

        client_config = ClientConfig(
            remote_server_addr=self.service.service_url, timeout=timeout_seconds
        )

        executor = FirefoxRemoteConnection(
            remote_server_addr=self.service.service_url,
            keep_alive=keep_alive,
            ignore_proxy=options._ignore_local_proxy,
            client_config=client_config,
        )

        super(FirefoxWebDriver, self).__init__(
            command_executor=executor, options=options
        )
        self._is_remote = False


# setting up webdrive and options for it
def setup_webdriver(timeout_seconds: int) -> webdriver.Firefox:
    try:
        print("Setting driver settings")
        firefox_options = Options()

        firefox_options.add_argument("-headless")

        firefox_options.profile = get_default_firefox_profile()

        print("setting up rest of the settings")
        firefox_options.set_preference("toolkit.startup.max_resumed_crashes", -1)
        firefox_options.set_preference("browser.sessionstore.resume_from_crash", False)
        firefox_options.set_preference("webdriver.load.strategy", "unstable")
        firefox_options.set_preference("browser.startup.homepage", "about:blank")
        firefox_options.set_preference("startup.homepage_welcome_url", "about:blank")
        firefox_options.set_preference("browser.startup.firstrunSkipsHomepage", True)
        firefox_options.set_preference("webdriver.firefox.timeout", timeout_seconds)

        print("getting webdriver")
        service = Service(executable_path=GeckoDriverManager().install())
        driver = CustomFirefoxWebDriver(
            timeout_seconds, options=firefox_options, service=service
        )

        driver.set_page_load_timeout(timeout_seconds)
        driver.set_script_timeout(timeout_seconds)
        driver.implicitly_wait(timeout_seconds)

        return driver
    except Exception as E:
        print("failed to start driver")
        print(f"Error: {str(E)}")
        exit()
