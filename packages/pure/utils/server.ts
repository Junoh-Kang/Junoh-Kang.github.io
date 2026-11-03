import { getCollection, type CollectionEntry, type CollectionKey } from 'astro:content'

export const prod = import.meta.env.PROD

/** Note: this function filters out draft posts based on the environment */
export async function getBlogCollection<T extends CollectionKey = 'blog'>(
  contentType: T = 'blog' as T
) {
  return await getCollection<T>(contentType, ({ data }) => {
    // Not in production & draft is not false
    return prod ? !data.draft : true
  })
}

function getYearFromCollection<T extends CollectionKey>(
  collection: CollectionEntry<T>
): number | undefined {
  const dateStr = collection.data.updatedDate ?? collection.data.publishDate
  return dateStr ? new Date(dateStr).getFullYear() : undefined
}
export function groupCollectionsByYear<T extends CollectionKey>(
  collections: CollectionEntry<T>[]
): [number, CollectionEntry<T>[]][] {
  const collectionsByYear = collections.reduce((acc, collection) => {
    const year = getYearFromCollection(collection)
    if (year !== undefined) {
      if (!acc.has(year)) {
        acc.set(year, [])
      }
      acc.get(year)?.push(collection)
    }
    return acc
  }, new Map<number, CollectionEntry<T>[]>())

  return Array.from(
    collectionsByYear.entries() as IterableIterator<[number, CollectionEntry<T>[]]>
  ).sort((a, b) => b[0] - a[0])
}

export function sortMDByDate<T extends CollectionKey>(
  collections: CollectionEntry<T>[]
): CollectionEntry<T>[] {
  return collections.sort((a, b) => {
    const aPin = 'pin' in a.data && a.data.pin === true
    const bPin = 'pin' in b.data && b.data.pin === true
    if (aPin !== bPin) {
      return Number(bPin) - Number(aPin)
    }

    const aDate = new Date(a.data.publishDate ?? 0).valueOf()
    const bDate = new Date(b.data.publishDate ?? 0).valueOf()
    return bDate - aDate
  })
}

/** Count posts per tag, showing frequently used tags first. */
export function getTagsWithCount(collections: CollectionEntry<'blog'>[]): [string, number][] {
  const counts = new Map<string, number>()
  for (const post of collections) {
    for (const tag of post.data.tags) counts.set(tag, (counts.get(tag) ?? 0) + 1)
  }
  return [...counts].sort(([a, countA], [b, countB]) => countB - countA || a.localeCompare(b))
}
