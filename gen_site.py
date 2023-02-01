import asyncio
import datetime
import json
import os

import pycountry
from jinja2 import Environment, FileSystemLoader
from twitchAPI.twitch import Twitch

import settings

if __name__ == '__main__':
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("index.html")

    with open("data.json", "r") as infile:
        data = json.load(infile)


    async def get_profile_images():
        twitch = await Twitch(settings.TWITCH_APP_ID, settings.TWITCH_APP_SECRET)
        streamer_names = [x["streamer"]["name"] for x in data]
        if not streamer_names:
            return {}

        users = twitch.get_users(logins=streamer_names)

        info = {}
        async for x in users:
            info[x.login] = x.profile_image_url
        return info


    info = asyncio.run(get_profile_images())

    for streamer in data:
        streamer["streamer"]["profile_image_url"] = info[streamer["streamer"]["name"]]

        language = pycountry.languages.get(alpha_2=streamer["streamer"]["language"])
        streamer["streamer"]["language_name"] = language.name

    content = template.render(data=data, now=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"))
    with open("output/index_tmp.html", "w") as outfile:
        outfile.write(content)

    with open("data.json", "w") as outfile:
        outfile.write(json.dumps(data))

    os.rename("output/index_tmp.html", "output/index.html")
