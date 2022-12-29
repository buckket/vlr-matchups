import asyncio
import datetime
import json
import os

import twitchAPI.helper
from jinja2 import Environment, FileSystemLoader
from twitchAPI.twitch import Twitch

import settings


async def update_streamer_info(login_name):
    twitch = await Twitch(settings.TWITCH_APP_ID, settings.TWITCH_APP_SECRET)
    streamer = await twitchAPI.helper.first(twitch.get_users(logins=[login_name]))
    return streamer.profile_image_url


if __name__ == '__main__':
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("index.html")

    with open("data.json", "r") as infile:
        data = json.load(infile)

    for streamer in data:
        profile_image_url = asyncio.run(update_streamer_info(streamer["streamer"]["name"]))
        streamer["streamer"]["profile_image_url"] = profile_image_url

    content = template.render(data=data, now=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"))
    with open("output/index_tmp.html", "w") as outfile:
        outfile.write(content)

    os.rename("output/index_tmp.html", "output/index.html")
