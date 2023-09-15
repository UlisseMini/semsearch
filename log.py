import discord
import asyncpg
import traceback
import os
import json
from dotenv import load_dotenv

load_dotenv()

tok = os.environ['DISCORD_TOKEN']
db_url = os.environ['DATABASE_URL']

bot = discord.Bot(intents=discord.Intents.all())


async def itable():
    conn = await asyncpg.connect(db_url)

    await conn.execute('''
        CREATE TABLE IF NOT EXISTS discord (
            id SERIAL PRIMARY KEY NOT NULL,
            message_id BIGINT NOT NULL,
            channel_id BIGINT NOT NULL,
            server_id BIGINT NOT NULL,
            timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            author TEXT NOT NULL,
            author_id BIGINT NOT NULL,
            content TEXT NOT NULL,
            reply_content TEXT,
            embed_json TEXT,
            embedding VECTOR(768)
        )
    ''')
    await conn.close()


async def log(message, conn):
    message_id = message.id
    channel_id = message.channel.id
    server_id = message.guild.id
    timestamp = message.created_at.replace(tzinfo=None)
    author = message.author.nick if hasattr(message.author, 'nick') else message.author.display_name
    author_id = message.author.id
    text_content = message.clean_content
    embed_json = json.dumps([e.to_dict() for e in message.embeds])

    reply_content = None
    if message.reference and message.reference.resolved:
        if isinstance(message.reference.resolved, discord.DeletedReferencedMessage):
            reply_content = '[deleted]'
        else:
            reply_content = message.reference.resolved.clean_content

    await conn.execute(
        'INSERT INTO discord(message_id, channel_id, server_id, timestamp, author, author_id, content, reply_content, embed_json) VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9)',
        message_id, channel_id, server_id, timestamp, author, author_id, text_content, reply_content, embed_json)


async def fetch(guild, channel):
    conn = await asyncpg.connect(db_url)

    lastm = None
    try:
        lastm_id = await conn.fetchval('SELECT message_id FROM discord WHERE channel_id = $1 AND server_id = $2 ORDER BY timestamp DESC LIMIT 1', channel.id, guild.id)
        lastm = await channel.fetch_message(lastm_id)
    except:
        print('no last message found')

    messages = await channel.history(limit=None, after=lastm, oldest_first=True).flatten() if lastm else await channel.history(limit=None, oldest_first=True).flatten()
    
    for message in messages:
        try:
            await log(message, conn)
        except Exception as e:
            traceback.print_exc()

    print(f'Logged {len(messages)} messages for channel {channel.name} in guild {guild.name}')

    await conn.close()


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
    await itable()

    try:
        for guild in bot.guilds:
            if guild.id == 1014436790251290624:
                for channel in guild.text_channels:
                    await fetch(guild, channel)
    except Exception as e:
        traceback.print_exc()
        print(e)


@bot.slash_command(guild_ids=[1014436790251290624])
async def search(ctx, query: str):
    conn = await asyncpg.connect(db_url)
    results = await conn.fetch('SELECT * FROM discord WHERE content LIKE $1 ORDER BY timestamp DESC LIMIT 10', f'%{query}%')

    await conn.close()

    if not results:
        await ctx.respond('No results found')
        return

    embed = discord.Embed(title="Search results")
    embed.add_field(name="Query", value=query, inline=False)

    def jump_url(result):
        return f'https://discord.com/channels/{result["server_id"]}/{result["channel_id"]}/{result["message_id"]}'

    for result in results:
        author, content, url = result['author'], result['content'], jump_url(result)
        embed.add_field(name=author, value=f'[{content}]({url})', inline=False)

    await ctx.respond(embed=embed)


@bot.event
async def on_message(message):
    conn = await asyncpg.connect(db_url)

    try:
        await log(message, conn)
    except Exception as e:
        traceback.print_exc()

    await conn.close()


if __name__ == "__main__":
    bot.run(tok)

